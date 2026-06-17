import os
import sys
import io
import json
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import get_session, SolveRecord
from pinn_solver import PINNSolver, create_ic_func, create_bc_func
from export_utils import export_csv, export_vtk, export_vtk_3d

app = Flask(__name__)
CORS(app)

active_solvers = {}


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'service': 'PINN Solver API'})


@app.route('/api/pde-types', methods=['GET'])
def get_pde_types():
    return jsonify({
        'types': [
            {
                'id': 'heat',
                'name': 'Heat Equation',
                'latex': r'\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}',
                'description': 'Heat diffusion equation',
                'default_params': {'alpha': 0.01}
            },
            {
                'id': 'wave',
                'name': 'Wave Equation',
                'latex': r'\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}',
                'description': 'Wave propagation equation',
                'default_params': {'c': 1.0}
            },
            {
                'id': 'elliptic',
                'name': 'Poisson/Laplace Equation',
                'latex': r'\nabla^2 u = f(x)',
                'description': 'Steady-state elliptic equation',
                'default_params': {'alpha': 1.0}
            },
            {
                'id': 'general',
                'name': 'General PDE',
                'latex': r'F(u, u_x, u_t, u_{xx}, u_{tt}) = 0',
                'description': 'Custom general PDE',
                'default_params': {'alpha': 0.01}
            }
        ]
    })


@app.route('/api/solve', methods=['POST'])
def solve_pde():
    data = request.get_json()
    pde_type = data.get('pde_type', 'heat')
    equation_latex = data.get('equation_latex', '')
    ic_expr = data.get('ic_expression', 'sin(pi*x)')
    bc_left_expr = data.get('bc_left_expression', '0')
    bc_right_expr = data.get('bc_right_expression', '0')
    domain = data.get('domain', {'x_min': 0.0, 'x_max': 1.0, 't_min': 0.0, 't_max': 1.0})
    params = data.get('params', {'alpha': 0.01})
    layers = data.get('layers', [2, 64, 64, 64, 1])
    N_f = data.get('n_collocation', 10000)
    N_i = data.get('n_initial', 200)
    N_b = data.get('n_boundary', 200)
    epochs = data.get('epochs', 5000)
    lr = data.get('learning_rate', 1e-3)
    name = data.get('name', '')

    try:
        solver = PINNSolver(
            pde_type=pde_type,
            domain={
                'x': (domain['x_min'], domain['x_max']),
                't': (domain['t_min'], domain['t_max'])
            },
            layers=layers,
            params=params
        )

        ic_func = create_ic_func(ic_expr, params) if ic_expr else None
        bc_left_func = create_bc_func(bc_left_expr, params) if bc_left_expr else None
        bc_right_func = create_bc_func(bc_right_expr, params) if bc_right_expr else None
        bc_funcs = (bc_left_func, bc_right_func) if (bc_left_func and bc_right_func) else None

        history = solver.train(
            ic_func=ic_func,
            bc_funcs=bc_funcs,
            N_f=N_f, N_i=N_i, N_b=N_b,
            epochs=epochs,
            learning_rate=lr,
            verbose=True
        )

        x, t, U = solver.predict_grid(nx=100, nt=100)
        final_loss = history[-1]['loss'] if history else None

        session = get_session()
        record = SolveRecord(
            name=name or f"{pde_type}_{len(active_solvers)}",
            pde_type=pde_type,
            equation_latex=equation_latex,
            ic_expression=ic_expr,
            bc_left_expression=bc_left_expr,
            bc_right_expression=bc_right_expr,
            domain_x_min=domain['x_min'],
            domain_x_max=domain['x_max'],
            domain_t_min=domain['t_min'],
            domain_t_max=domain['t_max'],
            params_json=json.dumps(params),
            layers_json=json.dumps(layers),
            n_collocation=N_f,
            n_initial=N_i,
            n_boundary=N_b,
            epochs=epochs,
            learning_rate=lr,
            final_loss=final_loss,
            training_history_json=json.dumps(history[-100:]),
            model_weights_json=json.dumps(solver.get_weights())
        )
        session.add(record)
        session.commit()
        record_id = record.id
        session.close()

        active_solvers[record_id] = solver

        grid_data = {
            'x': x.tolist(),
            't': t.tolist(),
            'u': U.tolist()
        }

        return jsonify({
            'id': record_id,
            'status': 'completed',
            'final_loss': final_loss,
            'training_history': history[-100:],
            'grid_data': grid_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/<int:record_id>', methods=['POST'])
def predict(record_id):
    data = request.get_json() or {}
    nx = data.get('nx', 100)
    nt = data.get('nt', 100)

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

    x, t, U = solver.predict_grid(nx=nx, nt=nt)
    return jsonify({
        'x': x.tolist(),
        't': t.tolist(),
        'u': U.tolist()
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

    x, t, U = solver.predict_grid(nx=100, nt=100)

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
    app.run(host='0.0.0.0', port=5001, debug=True)
