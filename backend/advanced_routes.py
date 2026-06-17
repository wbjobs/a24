"""新增高级功能路由：方程库、超参搜索、数值对比、分布式训练"""
import os
import sys
import time
import json
import threading
import queue
import numpy as np
from flask import request, jsonify, Response, stream_with_context, Blueprint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from pinn_solver_nd import NDPINNSolver, PINNSolver, create_ic_func_nd
    HAS_ND = True
except Exception as e:
    HAS_ND = False
    print(f'[高级路由] pinn_solver_nd加载失败: {e}')

try:
    from pde_library import PDE_LIBRARY, get_library_index, get_case, create_case_functions
    HAS_LIB = True
except Exception as e:
    HAS_LIB = False
    print(f'[高级路由] pde_library加载失败: {e}')

try:
    from hyperparam_search import run_hyperparameter_search, is_optuna_available
    HAS_OPTUNA = True
except Exception as e:
    HAS_OPTUNA = False
    print(f'[高级路由] hyperparam_search加载失败: {e}')

try:
    from classical_solvers import FDMSolver, compare_solutions
    HAS_FDM = True
except Exception as e:
    HAS_FDM = False
    print(f'[高级路由] classical_solvers加载失败: {e}')

try:
    from horovod_distributed import is_horovod_available, train_distributed
    HAS_HVD = True
except Exception as e:
    HAS_HVD = False
    print(f'[高级路由] horovod加载失败: {e}')

advanced_bp = Blueprint('advanced', __name__, url_prefix='/api')

hyperopt_state = {
    'running': False,
    'result': None,
    'progress': [],
    'error': None,
    'thread': None,
}

compare_state = {}


def register_advanced_routes(app):
    """在主Flask app上注册高级路由（通过挂载蓝图前先添加单独路由）"""

    # ------------- 方程库路由 -------------

    @app.route('/api/library', methods=['GET'])
    def library_index():
        if not HAS_LIB:
            return jsonify({'error': '方程库模块未加载', 'cases': []}), 200
        category = request.args.get('category', None)
        difficulty = request.args.get('difficulty', None)
        pde_type = request.args.get('pde_type', None)
        cases = get_library_index()
        if category:
            cases = [c for c in cases if c['category'] == category]
        if difficulty:
            cases = [c for c in cases if c['difficulty'] == difficulty or str(c['difficulty']).startswith(str(difficulty))]
        if pde_type:
            cases = [c for c in cases if c['pde_type'] == pde_type]
        categories = sorted(set(c['category'] for c in get_library_index()))
        pde_types = sorted(set(c['pde_type'] for c in get_library_index()))
        return jsonify({
            'total': len(get_library_index()),
            'filtered': len(cases),
            'categories': categories,
            'pde_types': pde_types,
            'cases': cases,
        })

    @app.route('/api/library/<case_id>', methods=['GET'])
    def library_case_detail(case_id):
        if not HAS_LIB:
            return jsonify({'error': '方程库模块未加载'}), 404
        case = get_case(case_id)
        if case is None:
            return jsonify({'error': f'案例不存在: {case_id}'}), 404
        return jsonify(case)

    @app.route('/api/library/<case_id>/solve', methods=['POST'])
    def library_case_solve(case_id):
        global sse_queues
        if not HAS_LIB or not HAS_ND:
            return jsonify({'error': '必要模块未加载'}), 500
        case = get_case(case_id)
        if case is None:
            return jsonify({'error': f'案例不存在: {case_id}'}), 404

        data = request.get_json() or {}
        override_epochs = data.get('epochs', case.get('default_epochs', 5000))
        override_layers = data.get('layers', case.get('default_layers'))
        use_fourier = data.get('use_fourier', True)
        use_adaptive = data.get('use_adaptive_activation', True)
        use_hc = data.get('use_hard_constraint', True)
        use_mc = data.get('use_mc_dropout', True)
        fb = data.get('fourier_bands', case.get('fourier_bands', [0.01, 0.1, 1, 10]))
        ff = data.get('fourier_freqs', 32)
        dr = data.get('dropout_rate', 0.05)
        mc = data.get('mc_samples', 30)
        li = data.get('log_interval', 50)
        nf = data.get('n_collocation', 10000)
        run_fdm_compare = data.get('run_fdm_compare', case.get('has_fdm_reference', False))
        do_hp_search = data.get('run_hyperopt', False)

        n_dims = case.get('n_dims', 1)
        ptype = case['pde_type']
        spatial_keys = ['x', 'y', 'z'][:n_dims]
        domain = {k: case['domain'][k] for k in spatial_keys + ['t']}
        params = case['params']
        try:
            ic_fn, bc_fns = create_case_functions(case)
        except Exception as e:
            return jsonify({'error': f'创建IC/BC函数失败: {e}'}), 400

        solver_args = {
            'pde_type': ptype,
            'n_dims': n_dims,
            'domain': domain,
            'layers': override_layers,
            'params': params,
            'use_fourier': use_fourier,
            'use_adaptive_activation': use_adaptive,
            'use_hard_constraint': use_hc,
            'use_mc_dropout': use_mc,
            'fourier_bands': fb,
            'fourier_freqs': ff,
            'dropout_rate': dr,
        }

        train_kwargs = {
            'ic_func': ic_fn,
            'bc_funcs': bc_fns,
            'N_f': nf,
            'N_i': data.get('n_initial', 500),
            'N_b': data.get('n_boundary', 500),
            'epochs': override_epochs,
            'learning_rate': data.get('learning_rate', 1e-3),
            'verbose': True,
            'log_interval': li,
        }

        task_id = f'task_lib_{case_id}_{int(time.time())}'

        def solve_thread():
            global sse_queues, active_solvers
            try:
                t0 = time.time()

                # 超参数搜索
                if do_hp_search and HAS_OPTUNA:
                    n_trials = data.get('hp_n_trials', 10)
                    quick_epochs = data.get('hp_quick_epochs', 200)

                    def _hp_cb(trial_i, best_p, best_l):
                        for qid in list(sse_queues.keys()):
                            q = sse_queues.get(qid)
                            if q is not None:
                                try:
                                    q.put({
                                        'type': 'hyperopt_progress',
                                        'task_id': task_id,
                                        'data': {
                                            'trial': trial_i,
                                            'best_loss': best_l,
                                            'best_params': best_p,
                                        }
                                    })
                                except Exception:
                                    pass

                    hp_res = run_hyperparameter_search(
                        pde_type=ptype,
                        n_dims=n_dims,
                        domain=domain,
                        params=params,
                        ic_func=ic_fn,
                        bc_funcs=bc_fns,
                        n_trials=n_trials,
                        quick_epochs=quick_epochs,
                        verbose=True,
                        progress_cb=_hp_cb,
                    )
                    best_params = hp_res['best_params']
                    if 'layers' in best_params:
                        solver_args['layers'] = best_params['layers']
                    if 'learning_rate' in best_params:
                        train_kwargs['learning_rate'] = best_params['learning_rate']
                    if 'n_collocation' in best_params:
                        train_kwargs['N_f'] = best_params['n_collocation']

                    for qid in list(sse_queues.keys()):
                        q = sse_queues.get(qid)
                        if q is not None:
                            try:
                                q.put({
                                    'type': 'hyperopt_complete',
                                    'task_id': task_id,
                                    'data': {
                                        'best_loss': hp_res['best_loss'],
                                        'best_params': best_params,
                                        'all_trials': hp_res['all_trials'],
                                    }
                                })
                            except Exception:
                                pass

                solver = NDPINNSolver(**solver_args)

                def _progress_cb(record):
                    for qid in list(sse_queues.keys()):
                        q = sse_queues.get(qid)
                        if q is not None:
                            try:
                                q.put({'type': 'progress', 'task_id': task_id, 'data': record})
                            except Exception:
                                pass

                solver.set_update_callback(_progress_cb)

                history = solver.train(**train_kwargs)
                train_time = time.time() - t0

                grid_sizes = {}
                for k in spatial_keys:
                    grid_sizes[k] = 50
                grid_sizes['t'] = 50
                grid_data = solver.predict_grid(grid_sizes, mc_samples=mc if use_mc else 1)
                grid_data['training_time_sec'] = train_time

                # FDM对比
                comparison = None
                if run_fdm_compare and HAS_FDM:
                    try:
                        fdm_solver = FDMSolver()
                        ptype_case = ptype
                        ic_case = case['ic_expression']
                        bl = case.get('bc_left_expression', '0')
                        br = case.get('bc_right_expression', '0')
                        ic_fn_eval = create_ic_func_nd(ic_case, params, ['x'])
                        bl_fn = create_ic_func_nd(bl, params, ['t'])
                        br_fn = create_ic_func_nd(br, params, ['t'])

                        fdm_args = {}
                        if ptype_case == 'heat':
                            fdm_result = fdm_solver.solve_heat_1d(
                                lambda x: ic_fn_eval(x=x),
                                lambda t: bl_fn(t=t),
                                lambda t: br_fn(t=t),
                                domain.get('x', (0,1))[0],
                                domain.get('x', (0,1))[1],
                                domain.get('t', (0,1))[0],
                                domain.get('t', (0,1))[1],
                                nx=200, nt=10000,
                                alpha=params.get('alpha', 0.01),
                            )
                        elif ptype_case == 'wave':
                            idt = case.get('ic_dt_expression', '0')
                            idt_fn = create_ic_func_nd(idt, params, ['x'])
                            fdm_result = fdm_solver.solve_wave_1d(
                                lambda x: ic_fn_eval(x=x),
                                lambda x: idt_fn(x=x),
                                lambda t: bl_fn(t=t),
                                lambda t: br_fn(t=t),
                                domain.get('x', (0,1))[0],
                                domain.get('x', (0,1))[1],
                                domain.get('t', (0,1))[0],
                                domain.get('t', (0,1))[1],
                                nx=200, nt=2000,
                                c=params.get('c', 1.0),
                            )
                        elif ptype_case == 'burgers':
                            fdm_result = fdm_solver.solve_burgers_1d(
                                lambda x: ic_fn_eval(x=x),
                                lambda t: bl_fn(t=t),
                                lambda t: br_fn(t=t),
                                domain.get('x', (0,1))[0],
                                domain.get('x', (0,1))[1],
                                domain.get('t', (0,1))[0],
                                domain.get('t', (0,1))[1],
                                nx=400, nt=20000,
                                nu=params.get('nu', 0.01),
                            )
                        elif ptype_case == 'allen_cahn':
                            fdm_result = fdm_solver.solve_allen_cahn_1d(
                                lambda x: ic_fn_eval(x=x),
                                lambda t: bl_fn(t=t),
                                lambda t: br_fn(t=t),
                                domain.get('x', (0,1))[0],
                                domain.get('x', (0,1))[1],
                                domain.get('t', (0,1))[0],
                                domain.get('t', (0,1))[1],
                                nx=400, nt=10000,
                                epsilon=params.get('epsilon', 0.01),
                                gamma=params.get('gamma', 1.0),
                            )
                        elif ptype_case == 'kdv':
                            fdm_result = fdm_solver.solve_kdv_1d(
                                lambda x: ic_fn_eval(x=x),
                                lambda t: bl_fn(t=t),
                                lambda t: br_fn(t=t),
                                domain.get('x', (-1,2))[0],
                                domain.get('x', (-1,2))[1],
                                domain.get('t', (0,1))[0],
                                domain.get('t', (0,1))[1],
                                nx=800, nt=40000,
                                alpha=params.get('alpha', 1.0),
                                beta=params.get('beta', 6.0),
                            )
                        else:
                            fdm_result = {'error': f'No FDM for pde_type={ptype_case}'}

                        if 'error' not in fdm_result:
                            comparison = compare_solutions(grid_data, fdm_result)
                            comparison['fdm_info'] = {
                                'method': fdm_result.get('method', ''),
                                'nx': fdm_result.get('nx', ''),
                                'nt': fdm_result.get('nt', ''),
                                'elapsed_sec': fdm_result.get('elapsed_sec', None),
                            }
                    except Exception as e:
                        comparison = {'error': str(e)}

                from app import get_session, SolveRecord, active_solvers
                from sqlalchemy.orm import Session as _S
                sess = get_session()
                record = SolveRecord(
                    name=data.get('name') or f"{case['name']}_{int(time.time())}",
                    pde_type=ptype,
                    equation_latex=case['equation_latex'],
                    ic_expression=case['ic_expression'],
                    bc_left_expression=case.get('bc_left_expression', '0'),
                    bc_right_expression=case.get('bc_right_expression', '0'),
                    domain_x_min=domain.get('x', (0,1))[0],
                    domain_x_max=domain.get('x', (0,1))[1],
                    domain_t_min=domain.get('t', (0,1))[0],
                    domain_t_max=domain.get('t', (0,1))[1],
                    params_json=json.dumps(params),
                    layers_json=json.dumps(solver_args['layers']),
                    n_collocation=train_kwargs['N_f'],
                    n_initial=train_kwargs['N_i'],
                    n_boundary=train_kwargs['N_b'],
                    epochs=override_epochs,
                    learning_rate=train_kwargs['learning_rate'],
                    final_loss=history[-1]['loss'] if history else None,
                    training_history_json=json.dumps(history[-200:]),
                    model_weights_json=json.dumps(solver.get_weights()),
                )
                sess.add(record)
                sess.commit()
                record_id = record.id
                sess.close()

                active_solvers[record_id] = solver

                result_msg = {
                    'type': 'complete',
                    'task_id': task_id,
                    'data': {
                        'id': record_id,
                        'case_id': case_id,
                        'status': 'completed',
                        'final_loss': history[-1]['loss'] if history else None,
                        'training_history': history[-200:],
                        'grid_data': grid_data,
                        'final_epochs': len(history),
                        'training_time_sec': train_time,
                        'comparison': comparison,
                    }
                }
                for qid in list(sse_queues.keys()):
                    q = sse_queues.get(qid)
                    if q is not None:
                        try:
                            q.put(result_msg)
                        except Exception:
                            pass
                compare_state[task_id] = comparison

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f'[LibrarySolve ERROR] {e}\n{tb}')
                for qid in list(sse_queues.keys()):
                    q = sse_queues.get(qid)
                    if q is not None:
                        try:
                            q.put({
                                'type': 'error',
                                'task_id': task_id,
                                'data': {'error': str(e), 'traceback': tb},
                            })
                        except Exception:
                            pass

        thread = threading.Thread(target=solve_thread, daemon=True)
        thread.start()

        return jsonify({
            'task_id': task_id,
            'case_id': case_id,
            'status': 'started',
            'message': f'方程库案例 {case["name"]} 求解已启动',
        })

    # ------------- 超参数搜索路由 -------------

    @app.route('/api/hyperopt/start', methods=['POST'])
    def start_hyperopt():
        global hyperopt_state
        if not HAS_OPTUNA:
            return jsonify({'error': 'Optuna未安装，请先 pip install optuna'}), 400
        if hyperopt_state['running']:
            return jsonify({'error': '已有超参搜索在运行中'}), 409

        data = request.get_json() or {}
        ptype = data.get('pde_type', 'heat')
        n_dims = data.get('n_dims', 1)
        spatial_keys = ['x', 'y', 'z'][:n_dims]
        n_trials = data.get('n_trials', 15)
        quick_epochs = data.get('quick_epochs', 200)

        domain = data.get('domain', {k: (0, 1) for k in spatial_keys})
        domain['t'] = data.get('t_domain', (0, 1))
        params = data.get('params', {})
        ic_expr = data.get('ic_expression', 'sin(pi*x)')
        bl_expr = data.get('bc_left_expression', '0')
        br_expr = data.get('bc_right_expression', '0')

        from pinn_solver_nd import create_ic_func_nd
        try:
            ic_fn = create_ic_func_nd(ic_expr, params, spatial_keys)
            bc_fns = [
                create_ic_func_nd(bl_expr, params, ['t']),
                create_ic_func_nd(br_expr, params, ['t']),
            ]
        except Exception as e:
            return jsonify({'error': f'IC/BC表达式错误: {e}'}), 400

        hyperopt_state['running'] = True
        hyperopt_state['result'] = None
        hyperopt_state['progress'] = []
        hyperopt_state['error'] = None

        def cb(trial_i, best_p, best_l):
            hyperopt_state['progress'].append({
                'trial': trial_i,
                'best_loss': best_l,
                'best_params': best_p,
                'time': time.time(),
            })

        def run_thread():
            global hyperopt_state
            try:
                res = run_hyperparameter_search(
                    pde_type=ptype,
                    n_dims=n_dims,
                    domain=domain,
                    params=params,
                    ic_func=ic_fn,
                    bc_funcs=bc_fns,
                    n_trials=n_trials,
                    quick_epochs=quick_epochs,
                    verbose=True,
                    progress_cb=cb,
                )
                hyperopt_state['result'] = {
                    k: v for k, v in res.items() if k != 'study'
                }
                hyperopt_state['running'] = False
            except Exception as e:
                hyperopt_state['running'] = False
                hyperopt_state['error'] = str(e)

        hyperopt_state['thread'] = threading.Thread(target=run_thread, daemon=True)
        hyperopt_state['thread'].start()

        return jsonify({'status': 'started', 'n_trials': n_trials, 'pde_type': ptype})

    @app.route('/api/hyperopt/status', methods=['GET'])
    def hyperopt_status():
        return jsonify({
            'running': hyperopt_state['running'],
            'progress_count': len(hyperopt_state['progress']),
            'progress': hyperopt_state['progress'][-20:],
            'result': hyperopt_state['result'],
            'error': hyperopt_state['error'],
        })

    # ------------- 数值方法对比路由 -------------

    @app.route('/api/fdm/solve', methods=['POST'])
    def fdm_solve():
        if not HAS_FDM:
            return jsonify({'error': 'FDM求解器未加载'}), 500
        data = request.get_json() or {}
        ptype = data.get('pde_type', 'heat')
        from pinn_solver_nd import create_ic_func_nd
        params = data.get('params', {})
        bl_fn = create_ic_func_nd(data.get('bc_left', '0'), params, ['t'])
        br_fn = create_ic_func_nd(data.get('bc_right', '0'), params, ['t'])
        domain = data.get('domain', {'x': (0, 1), 't': (0, 1)})

        def bcl(t_val):
            return bl_fn(t=t_val)

        def bcr(t_val):
            return br_fn(t=t_val)

        fdm_solver = FDMSolver()
        if ptype == 'heat':
            ic_fn = create_ic_func_nd(data.get('ic', 'sin(pi*x)'), params, ['x'])
            result = fdm_solver.solve_heat_1d(
                lambda x: ic_fn(x=x), bcl, bcr,
                domain['x'][0], domain['x'][1],
                domain['t'][0], domain['t'][1],
                nx=data.get('nx', 200), nt=data.get('nt', 10000),
                alpha=params.get('alpha', 0.01),
            )
        elif ptype == 'wave':
            ic_fn = create_ic_func_nd(data.get('ic', 'sin(pi*x)'), params, ['x'])
            idt_fn = create_ic_func_nd(data.get('ic_dt', '0'), params, ['x'])
            result = fdm_solver.solve_wave_1d(
                lambda x: ic_fn(x=x), lambda x: idt_fn(x=x), bcl, bcr,
                domain['x'][0], domain['x'][1],
                domain['t'][0], domain['t'][1],
                nx=data.get('nx', 200), nt=data.get('nt', 2000),
                c=params.get('c', 1.0),
            )
        elif ptype == 'burgers':
            ic_fn = create_ic_func_nd(data.get('ic', '-sin(pi*x)'), params, ['x'])
            result = fdm_solver.solve_burgers_1d(
                lambda x: ic_fn(x=x), bcl, bcr,
                domain['x'][0], domain['x'][1],
                domain['t'][0], domain['t'][1],
                nx=data.get('nx', 400), nt=data.get('nt', 20000),
                nu=params.get('nu', 0.01),
            )
        else:
            result = fdm_solver.solve(ptype, lambda x: 0, lambda t: 0, lambda t: 0)

        # 转可JSON化
        for k in ['x', 't', 'y', 'u']:
            if k in result and hasattr(result[k], 'tolist'):
                result[k] = result[k].tolist()
        return jsonify(result)

    @app.route('/api/compare', methods=['POST'])
    def compare_api():
        if not HAS_FDM:
            return jsonify({'error': 'FDM求解器未加载'}), 500
        data = request.get_json() or {}
        record_id = data.get('record_id')
        grid_data = data.get('grid_data')
        fdm_result = data.get('fdm_result')

        if fdm_result is None and record_id:
            from app import active_solvers
            solver = active_solvers.get(record_id)
            if solver is not None:
                # 运行FDM
                pass

        if grid_data and fdm_result:
            comp = compare_solutions(grid_data, fdm_result)
            return jsonify(comp)
        return jsonify({'error': '需提供grid_data和fdm_result'}), 400

    # ------------- 分布式状态路由 -------------

    @app.route('/api/system/capabilities', methods=['GET'])
    def system_capabilities():
        return jsonify({
            'horovod_available': is_horovod_available() if HAS_HVD else False,
            'optuna_available': HAS_OPTUNA and is_optuna_available(),
            'fdm_available': HAS_FDM,
            'library_available': HAS_LIB,
            'nd_solver_available': HAS_ND,
            'library_cases': len(PDE_LIBRARY) if HAS_LIB else 0,
            'supported_pde_types': list(NDPINNSolver.EQUATION_TYPES.keys()) if HAS_ND else [],
        })
