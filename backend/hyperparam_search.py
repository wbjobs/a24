"""Optuna自动超参数搜索模块"""
import os
import time
import json
import numpy as np

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    optuna = None
    HAS_OPTUNA = False


def is_optuna_available():
    return HAS_OPTUNA


EQUATION_HP_RANGES = {
    'heat': {
        'layers_base': [32, 64, 128, 256],
        'n_layers': [2, 3, 4, 5, 6],
        'lr_range': [1e-4, 1e-2],
        'n_collocation_range': [5000, 30000],
    },
    'wave': {
        'layers_base': [64, 128, 256],
        'n_layers': [3, 4, 5, 6, 7],
        'lr_range': [5e-4, 5e-3],
        'n_collocation_range': [10000, 50000],
    },
    'burgers': {
        'layers_base': [32, 64, 128, 256],
        'n_layers': [3, 4, 5, 6],
        'lr_range': [5e-4, 2e-3],
        'n_collocation_range': [10000, 40000],
    },
    'ns_incompressible': {
        'layers_base': [128, 256, 512],
        'n_layers': [5, 6, 7, 8],
        'lr_range': [1e-4, 1e-3],
        'n_collocation_range': [30000, 100000],
    },
    'allen_cahn': {
        'layers_base': [64, 128, 256],
        'n_layers': [4, 5, 6, 7],
        'lr_range': [3e-4, 3e-3],
        'n_collocation_range': [10000, 40000],
    },
    'kdv': {
        'layers_base': [64, 128, 256, 512],
        'n_layers': [4, 5, 6, 7],
        'lr_range': [1e-4, 2e-3],
        'n_collocation_range': [15000, 60000],
    },
    'reaction_diffusion': {
        'layers_base': [64, 128, 256],
        'n_layers': [3, 4, 5, 6],
        'lr_range': [3e-4, 3e-3],
        'n_collocation_range': [10000, 50000],
    },
    'sine_gordon': {
        'layers_base': [64, 128, 256],
        'n_layers': [4, 5, 6],
        'lr_range': [1e-4, 2e-3],
        'n_collocation_range': [10000, 40000],
    },
    'schrodinger': {
        'layers_base': [128, 256, 512],
        'n_layers': [5, 6, 7, 8],
        'lr_range': [1e-4, 1e-3],
        'n_collocation_range': [20000, 80000],
    },
    'advection': {
        'layers_base': [32, 64, 128],
        'n_layers': [2, 3, 4],
        'lr_range': [1e-3, 5e-3],
        'n_collocation_range': [5000, 20000],
    },
    'euler': {
        'layers_base': [128, 256, 512],
        'n_layers': [5, 6, 7, 8],
        'lr_range': [1e-4, 1e-3],
        'n_collocation_range': [30000, 80000],
    },
    'default': {
        'layers_base': [32, 64, 128, 256],
        'n_layers': [3, 4, 5, 6],
        'lr_range': [5e-4, 5e-3],
        'n_collocation_range': [10000, 40000],
    },
}


def _get_range(pde_type):
    return EQUATION_HP_RANGES.get(pde_type, EQUATION_HP_RANGES['default'])


def build_solver_from_trial(trial, pde_type, n_dims=1, domain=None, params=None,
                            use_fourier=True, use_adaptive=True,
                            use_hard_constraint=True, use_mc_dropout=True,
                            fourier_bands=None, fourier_freqs=32,
                            dropout_rate=0.05):
    """根据Optuna trial构建solver参数"""
    from pinn_solver_nd import NDPINNSolver, create_ic_func_nd
    r = _get_range(pde_type)

    base_width = trial.suggest_categorical('hidden_width', r['layers_base'])
    n_layers = trial.suggest_int('n_layers', min(r['n_layers']), max(r['n_layers']))
    lr = trial.suggest_float('learning_rate', r['lr_range'][0], r['lr_range'][1], log=True)
    n_coll = trial.suggest_int('n_collocation', r['n_collocation_range'][0], r['n_collocation_range'][1])

    input_dim = n_dims + 1
    n_out = NDPINNSolver.OUTPUT_DIMS.get(pde_type, 1)
    layers = [input_dim] + [base_width] * n_layers + [n_out]

    solver_kwargs = {
        'pde_type': pde_type,
        'n_dims': n_dims,
        'domain': domain,
        'layers': layers,
        'params': params,
        'use_fourier': use_fourier,
        'use_adaptive_activation': use_adaptive,
        'use_hard_constraint': use_hard_constraint,
        'use_mc_dropout': use_mc_dropout,
        'fourier_bands': fourier_bands,
        'fourier_freqs': fourier_freqs,
        'dropout_rate': dropout_rate,
    }

    extra = {
        'learning_rate': lr,
        'n_collocation': n_coll,
    }
    return solver_kwargs, extra


def run_hyperparameter_search(
    pde_type='heat',
    n_dims=1,
    domain=None,
    params=None,
    ic_func=None,
    bc_funcs=None,
    n_trials=20,
    quick_epochs=200,
    quick_n_collocation=None,
    verbose=True,
    n_jobs=1,
    direction='minimize',
    progress_cb=None,
):
    """
    运行Optuna超参数搜索

    Args:
        pde_type: 方程类型
        n_dims: 空间维度
        domain: 求解域
        params: 物理参数
        ic_func: 初始条件函数
        bc_funcs: 边界条件函数列表
        n_trials: 尝试次数
        quick_epochs: 快速搜索的epoch数（实际使用中推荐）
        n_jobs: 并行job数
        direction: 'minimize' 或 'maximize'
        progress_cb: 进度回调 fn(trial_idx, best_params, best_loss)
    Returns:
        dict: best_params, best_loss, all_trials, study (optuna study object)
    """
    if not HAS_OPTUNA:
        raise ImportError('Optuna未安装。请先 pip install optuna')

    from pinn_solver_nd import NDPINNSolver
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=direction)
    spatial_keys = ['x', 'y', 'z'][:n_dims]
    if domain is None:
        domain = {}
        for k in spatial_keys:
            domain[k] = (0.0, 1.0)
        domain['t'] = (0.0, 1.0)

    rng = _get_range(pde_type)
    if quick_n_collocation is None:
        quick_n_collocation = max(2000, rng['n_collocation_range'][0] // 3)

    all_trials = []

    def objective(trial):
        try:
            solver_kwargs, extra = build_solver_from_trial(
                trial, pde_type, n_dims, domain, params,
                use_fourier=True, use_adaptive=True,
                use_hard_constraint=True, use_mc_dropout=False,
            )

            epochs_local = quick_epochs
            solver = NDPINNSolver(**solver_kwargs)
            hist = solver.train(
                ic_func=ic_func,
                bc_funcs=bc_funcs,
                N_f=extra['n_collocation'],
                N_i=max(100, solver_kwargs.get('layers', [0])[1] // 2),
                N_b=max(100, solver_kwargs.get('layers', [0])[1] // 2),
                epochs=epochs_local,
                learning_rate=extra['learning_rate'],
                verbose=False,
                log_interval=max(1, epochs_local // 10),
            )

            # 用最后30%的平均损失作为指标（避免尾部偶然波动）
            k = max(1, len(hist) // 3)
            last_losses = [h['loss'] for h in hist[-k:]]
            obj = float(np.mean(last_losses))
            final = hist[-1]['loss'] if hist else float('inf')

            all_trials.append({
                'number': trial.number,
                'params': trial.params,
                'avg_loss': obj,
                'final_loss': final,
                'layers': solver_kwargs['layers'],
            })

            if verbose:
                print(f'[Optuna] Trial #{trial.number}: lr={extra["learning_rate"]:.2e} '
                      f'width={trial.params["hidden_width"]} L={trial.params["n_layers"]} '
                      f'Nf={extra["n_collocation"]} -> loss={obj:.3e}')

            if progress_cb:
                try:
                    best = min([t['avg_loss'] for t in all_trials])
                    best_trial = [t for t in all_trials if t['avg_loss'] == best][-1]
                    progress_cb(len(all_trials), best_trial['params'], best)
                except Exception:
                    pass
            return obj
        except Exception as e:
            print(f'[Optuna] Trial #{trial.number} failed: {e}')
            return float('inf')

    study.optimize(objective, n_trials=n_trials, n_jobs=max(1, n_jobs))

    if study.best_trial is not None:
        best_params = study.best_trial.params
        best_loss = study.best_value
        width = best_params['hidden_width']
        nl = best_params['n_layers']
        n_out = NDPINNSolver.OUTPUT_DIMS.get(pde_type, 1)
        in_dim = n_dims + 1
        best_params['layers'] = [in_dim] + [width] * nl + [n_out]

        if verbose:
            print(f'\n[Optuna] 最佳结果: loss={best_loss:.3e}')
            for k, v in best_params.items():
                print(f'  {k} = {v}')
    else:
        best_params = {}
        best_loss = float('inf')

    return {
        'best_params': best_params,
        'best_loss': best_loss,
        'all_trials': all_trials,
        'study': study,
        'n_trials_done': len(all_trials),
    }
