import os
import sys
import io
import json
import time
import queue
import threading
import numpy as np
from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import get_session, SolveRecord
from pinn_solver import PINNSolver, create_ic_func, create_bc_func
from export_utils import export_csv, export_vtk, export_vtk_3d

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

active_solvers = {}
sse_queues = {}
_sse_counter = 0


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'service': 'PINN Solver API v2'})


@app.route('/api/pde-types', methods=['GET'])
def get_pde_types():
    return jsonify({
        'types': [
            {
                'id': 'heat',
                'name': '热传导方程',
                'latex': r'\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}',
                'description': '描述热扩散过程的抛物型方程',
                'default_params': {'alpha': 0.01}
            },
            {
                'id': 'wave',
                'name': '波动方程',
                'latex': r'\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}',
                'description': '描述波传播的双曲型方程',
                'default_params': {'c': 1.0}
            },
            {
                'id': 'elliptic',
                'name': '泊松/拉普拉斯方程',
                'latex': r'\alpha \nabla^2 u = f(x)',
                'description': '稳态椭圆型方程',
                'default_params': {'alpha': 1.0, 'f': 0.0}
            },
            {
                'id': 'general',
                'name': '通用PDE',
                'latex': r'F(u, u_x, u_t, u_{xx}, u_{tt}) = 0',
                'description': '自定义偏微分方程',
                'default_params': {'alpha': 0.01}
            }
        ]
    })


@app.route('/api/advanced-options', methods=['GET'])
def get_advanced_options():
    return jsonify({
        'use_fourier': True,
        'use_adaptive_activation': True,
        'use_hard_constraint': True,
        'use_mc_dropout': True,
        'fourier_bands': [0.01, 0.1, 1.0, 10.0],
        'fourier_freqs': 32,
        'dropout_rate': 0.05,
        'mc_samples': 30,
    })


def _solve_task(task_id, solver_args, ic_func, bc_funcs, params):
    global sse_queues
    try:
        solver = PINNSolver(**solver_args)

        def _progress_cb(record):
            global sse_queues
            for qid in list(sse_queues.keys()):
                q = sse_queues.get(qid)
                if q is not None:
                    try:
                        q.put({
                            'type': 'progress',
                            'task_id': task_id,
                            'data': record
                        })
                    except Exception:
                        pass

        solver.set_update_callback(_progress_cb)

        history = solver.train(
            ic_func=ic_func,
            bc_funcs=bc_funcs,
            N_f=params.get('n_collocation', 10000),
            N_i=params.get('n_initial', 500),
            N_b=params.get('n_boundary', 500),
            epochs=params.get('epochs', 5000),
            learning_rate=params.get('learning_rate', 1e-3),
            verbose=True,
            log_interval=params.get('log_interval', 20)
        )

        mc_samples = params.get('mc_samples', 30)
        use_mc = solver_args.get('use_mc_dropout', True) and mc_samples > 1

        pred_result = solver.predict_grid(nx=100, nt=100, mc_samples=mc_samples if use_mc else 1)

        if use_mc and len(pred_result) == 5:
            x, t, U_mean, U_std, U_unc = pred_result
            grid_data = {
                'x': x.tolist(),
                't': t.tolist(),
                'u': U_mean.tolist(),
                'u_std': U_std.tolist(),
                'u_uncertainty': U_unc.tolist(),
                'has_uncertainty': True,
            }
        else:
            if len(pred_result) == 3:
                x, t, U = pred_result
            else:
                x, t, U = pred_result[0], pred_result[1], pred_result[2]
            grid_data = {
                'x': x.tolist(),
                't': t.tolist(),
                'u': U.tolist(),
                'has_uncertainty': False,
            }

        final_loss = history[-1]['loss'] if history else None

        session = get_session()
        record = SolveRecord(
            name=params.get('name') or f"{solver_args['pde_type']}_{int(time.time())}",
            pde_type=solver_args['pde_type'],
            equation_latex=params.get('equation_latex', ''),
            ic_expression=params.get('ic_expression', ''),
            bc_left_expression=params.get('bc_left_expression', ''),
            bc_right_expression=params.get('bc_right_expression', ''),
            domain_x_min=solver_args['domain']['x'][0],
            domain_x_max=solver_args['domain']['x'][1],
            domain_t_min=solver_args['domain']['t'][0],
            domain_t_max=solver_args['domain']['t'][1],
            params_json=json.dumps(solver_args.get('params', {})),
            layers_json=json.dumps(solver_args.get('layers', [])),
            n_collocation=params.get('n_collocation', 10000),
            n_initial=params.get('n_initial', 500),
            n_boundary=params.get('n_boundary', 500),
            epochs=params.get('epochs', 5000),
            learning_rate=params.get('learning_rate', 1e-3),
            final_loss=final_loss,
            training_history_json=json.dumps(history[-200:]),
            model_weights_json=json.dumps(solver.get_weights())
        )
        session.add(record)
        session.commit()
        record_id = record.id
        session.close()

        active_solvers[record_id] = solver

        result_msg = {
            'type': 'complete',
            'task_id': task_id,
            'data': {
                'id': record_id,
                'status': 'completed',
                'final_loss': final_loss,
                'training_history': history[-200:],
                'grid_data': grid_data,
                'final_epochs': len(history),
            }
        }

        for qid in list(sse_queues.keys()):
            q = sse_queues.get(qid)
            if q is not None:
                try:
                    q.put(result_msg)
                except Exception:
                    pass

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Solve task error: {e}\n{tb}")
        for qid in list(sse_queues.keys()):
            q = sse_queues.get(qid)
            if q is not None:
                try:
                    q.put({
                        'type': 'error',
                        'task_id': task_id,
                        'data': {'error': str(e), 'traceback': tb}
                    })
                except Exception:
                    pass


@app.route('/api/solve', methods=['POST'])
def solve_pde():
    global _sse_counter, sse_queues
    data = request.get_json() or {}
    pde_type = data.get('pde_type', 'heat')
    equation_latex = data.get('equation_latex', '')
    ic_expr = data.get('ic_expression', 'sin(pi*x)')
    bc_left_expr = data.get('bc_left_expression', '0')
    bc_right_expr = data.get('bc_right_expression', '0')
    domain = data.get('domain', {'x_min': 0.0, 'x_max': 1.0, 't_min': 0.0, 't_max': 1.0})
    params = data.get('params', {'alpha': 0.01})
    layers = data.get('layers', [2, 128, 128, 128, 128, 1])
    N_f = data.get('n_collocation', 10000)
    N_i = data.get('n_initial', 500)
    N_b = data.get('n_boundary', 500)
    epochs = data.get('epochs', 5000)
    lr = data.get('learning_rate', 1e-3)
    name = data.get('name', '')
    log_interval = data.get('log_interval', 20)

    use_fourier = data.get('use_fourier', True)
    use_adaptive_activation = data.get('use_adaptive_activation', True)
    use_hard_constraint = data.get('use_hard_constraint', True)
    use_mc_dropout = data.get('use_mc_dropout', True)
    fourier_bands = data.get('fourier_bands', [0.01, 0.1, 1.0, 10.0])
    fourier_freqs = data.get('fourier_freqs', 32)
    dropout_rate = data.get('dropout_rate', 0.05)
    mc_samples = data.get('mc_samples', 30)

    _sse_counter += 1
    task_id = f"task_{_sse_counter}_{int(time.time())}"

    solver_args = {
        'pde_type': pde_type,
        'domain': {
            'x': (float(domain['x_min']), float(domain['x_max'])),
            't': (float(domain['t_min']), float(domain['t_max']))
        },
        'layers': layers,
        'params': params,
        'use_fourier': use_fourier,
        'use_adaptive_activation': use_adaptive_activation,
        'use_hard_constraint': use_hard_constraint,
        'use_mc_dropout': use_mc_dropout,
        'fourier_bands': fourier_bands,
        'fourier_freqs': fourier_freqs,
        'dropout_rate': dropout_rate,
    }

    train_params = {
        'n_collocation': N_f,
        'n_initial': N_i,
        'n_boundary': N_b,
        'epochs': epochs,
        'learning_rate': lr,
        'name': name,
        'equation_latex': equation_latex,
        'ic_expression': ic_expr,
        'bc_left_expression': bc_left_expr,
        'bc_right_expression': bc_right_expr,
        'log_interval': log_interval,
        'mc_samples': mc_samples,
    }

    ic_func = create_ic_func(ic_expr, params) if ic_expr else None
    bc_left_func = create_bc_func(bc_left_expr, params) if bc_left_expr else None
    bc_right_func = create_bc_func(bc_right_expr, params) if bc_right_expr else None
    bc_funcs = (bc_left_func, bc_right_func) if (bc_left_func and bc_right_func) else None

    thread = threading.Thread(
        target=_solve_task,
        args=(task_id, solver_args, ic_func, bc_funcs, train_params),
        daemon=True
    )
    thread.start()

    return jsonify({
        'task_id': task_id,
        'status': 'started',
        'message': '求解任务已启动，请通过SSE流获取进度',
    })


@app.route('/api/stream', methods=['GET'])
def stream_events():
    global sse_queues
    client_id = f"client_{int(time.time() * 1000)}_{os.urandom(4).hex()}"
    q = queue.Queue(maxsize=500)
    sse_queues[client_id] = q

    def generate():
        try:
            yield f"event: connected\ndata: {json.dumps({'client_id': client_id, 'status': 'connected'})}\n\n"
            last_heartbeat = time.time()
            while True:
                try:
                    msg = q.get(timeout=5.0)
                    event_name = msg.get('type', 'message')
                    payload = msg.get('data', {})
                    yield f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"
                    last_heartbeat = time.time()
                except queue.Empty:
                    if time.time() - last_heartbeat > 90:
                        break
                    yield f"event: heartbeat\ndata: {json.dumps({'ts': time.time()})}\n\n"
        except GeneratorExit:
            pass
        finally:
            sse_queues.pop(client_id, None)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        }
    )


@app.route('/api/predict/<int:record_id>', methods=['POST'])
def predict(record_id):
    data = request.get_json() or {}
    nx = data.get('nx', 100)
    nt = data.get('nt', 100)
    mc_samples = data.get('mc_samples', 30)
    compute_uncertainty = data.get('compute_uncertainty', True)

    solver = active_solvers.get(record_id)
    if solver is None:
        session = get_session()
        record = session.query(SolveRecord).get(record_id)
        if record is None:
            session.close()
            return jsonify({'error': 'Record not found'}), 404
        solver = PINNSolver(
            pde_type=record.pde_type,
            domain={
                'x': (record.domain_x_min, record.domain_x_max),
                't': (record.domain_t_min, record.domain_t_max)
            },
            layers=json.loads(record.layers_json),
            params=json.loads(record.params_json)
        )
        if record.model_weights_json:
            solver.set_weights(json.loads(record.model_weights_json))
        session.close()
        active_solvers[record_id] = solver

    use_mc = solver.use_mc_dropout and compute_uncertainty and mc_samples > 1
    pred = solver.predict_grid(nx=nx, nt=nt, mc_samples=mc_samples if use_mc else 1)

    if use_mc and len(pred) == 5:
        x, t, U_mean, U_std, U_unc = pred
        return jsonify({
            'x': x.tolist(),
            't': t.tolist(),
            'u': U_mean.tolist(),
            'u_std': U_std.tolist(),
            'u_uncertainty': U_unc.tolist(),
            'has_uncertainty': True,
        })
    else:
        if len(pred) == 3:
            x, t, U = pred
        else:
            x, t, U = pred[0], pred[1], pred[2]
        return jsonify({
            'x': x.tolist(),
            't': t.tolist(),
            'u': U.tolist(),
            'has_uncertainty': False,
        })


@app.route('/api/history', methods=['GET'])
def get_history():
    session = get_session()
    records = session.query(SolveRecord).order_by(SolveRecord.created_at.desc()).all()
    result = [r.to_dict() for r in records]
    session.close()
    return jsonify(result)


@app.route('/api/history/<int:record_id>', methods=['GET'])
def get_record(record_id):
    session = get_session()
    record = session.query(SolveRecord).get(record_id)
    if record is None:
        session.close()
        return jsonify({'error': 'Record not found'}), 404
    result = record.to_dict()
    session.close()
    return jsonify(result)


@app.route('/api/history/<int:record_id>', methods=['DELETE'])
def delete_record(record_id):
    session = get_session()
    record = session.query(SolveRecord).get(record_id)
    if record is None:
        session.close()
        return jsonify({'error': 'Record not found'}), 404
    session.delete(record)
    session.commit()
    session.close()
    if record_id in active_solvers:
        del active_solvers[record_id]
    return jsonify({'status': 'deleted'})


@app.route('/api/export/<int:record_id>/<format>', methods=['GET'])
def export_result(record_id, format):
    solver = active_solvers.get(record_id)
    if solver is None:
        session = get_session()
        record = session.query(SolveRecord).get(record_id)
        if record is None:
            session.close()
            return jsonify({'error': 'Record not found'}), 404
        solver = PINNSolver(
            pde_type=record.pde_type,
            domain={
                'x': (record.domain_x_min, record.domain_x_max),
                't': (record.domain_t_min, record.domain_t_max)
            },
            layers=json.loads(record.layers_json),
            params=json.loads(record.params_json)
        )
        if record.model_weights_json:
            solver.set_weights(json.loads(record.model_weights_json))
        session.close()
        active_solvers[record_id] = solver

    pred = solver.predict_grid(nx=100, nt=100, mc_samples=1)
    if len(pred) == 3:
        x, t, U = pred
    else:
        x, t, U = pred[0], pred[1], pred[2]

    if format == 'csv':
        content = export_csv(x, t, U)
        return send_file(
            io.BytesIO(content.encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'pinn_result_{record_id}.csv'
        )
    elif format == 'vtk':
        content = export_vtk(x, t, U)
        return send_file(
            io.BytesIO(content.encode()),
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=f'pinn_result_{record_id}.vtk'
        )
    elif format == 'vtk3d':
        content = export_vtk_3d(x, t, U)
        return send_file(
            io.BytesIO(content.encode()),
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=f'pinn_result_3d_{record_id}.vtk'
        )
    else:
        return jsonify({'error': f'Unsupported format: {format}'}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
