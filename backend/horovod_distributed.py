"""Horovod分布式多GPU训练封装"""
import os
import time
import numpy as np

try:
    import horovod.tensorflow as hvd
    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False


def is_horovod_available():
    return HAS_HOROVOD


def init_horovod():
    """初始化Horovod，返回hvd模块或None"""
    if not HAS_HOROVOD:
        print('[Horovod] Horovod未安装，将使用单机模式')
        return None
    try:
        hvd.init()
        print(f'[Horovod] rank={hvd.rank()}, size={hvd.size()}, local_rank={hvd.local_rank()}')
        return hvd
    except Exception as e:
        print(f'[Horovod] 初始化失败: {e}，使用单机模式')
        return None


def train_distributed(solver_class, solver_kwargs, train_kwargs,
                      use_horovod=True, verbose=True):
    """
    分布式训练入口
    Args:
        solver_class: 求解器类 (NDPINNSolver or PINNSolver)
        solver_kwargs: 构造求解器的关键字参数
        train_kwargs: train()的关键字参数 (含ic_func, bc_funcs等)
        use_horovod: 是否启用Horovod
    Returns:
        (solver实例, history, training_time_sec)
    """
    t0 = time.time()

    hvd_mod = None
    if use_horovod:
        hvd_mod = init_horovod()

    rank = hvd_mod.rank() if hvd_mod is not None else 0
    size = hvd_mod.size() if hvd_mod is not None else 1

    if hvd_mod is not None:
        solver_kwargs['n_dims'] = solver_kwargs.get('n_dims', 1)

    solver = solver_class(**solver_kwargs)

    if 'N_f' in train_kwargs:
        train_kwargs['N_f'] = max(100, train_kwargs['N_f'] // max(1, size))

    if rank == 0 and verbose:
        nd = solver_kwargs.get('n_dims', 1)
        print(f'[Distributed] {nd}D+1问题 · {size}进程 · '
              f'每进程配置点={train_kwargs.get("N_f","?")} · '
              f'epochs={train_kwargs.get("epochs","?")}')

    cb_original = train_kwargs.get('hvd', None)
    train_kwargs['hvd'] = hvd_mod

    history = solver.train(**train_kwargs)

    elapsed = time.time() - t0

    if rank == 0 and verbose:
        final_loss = history[-1]['loss'] if history else float('nan')
        print(f'[Distributed] 完成 · 用时={elapsed:.2f}s · final_loss={final_loss:.3e}')

    return solver, history, elapsed
