import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from pinn_solver import PINNSolver, create_ic_func, create_bc_func

params = {'alpha': 0.01}
ic_func = create_ic_func('sin(pi*x)', params)
bc_left = create_bc_func('0', params)
bc_right = create_bc_func('0', params)

solver_args = {
    'pde_type': 'heat',
    'domain': {'x': (0.0, 1.0), 't': (0.0, 1.0)},
    'layers': [2, 64, 64, 1],
    'params': params,
    'use_fourier': True,
    'use_adaptive_activation': True,
    'use_hard_constraint': True,
    'use_mc_dropout': True,
    'fourier_bands': [0.01, 0.1, 1, 10],
    'fourier_freqs': 32,
    'dropout_rate': 0.05,
}

t0 = time.time()
print('Creating solver...')
s = PINNSolver(**solver_args)
print('Build OK, took', time.time() - t0, 's')

def cb(rec):
    global cb_count
    cb_count = (cb_count + 1) if 'cb_count' in globals() else 1
    if cb_count % 5 == 0:
        print(f"  cb epoch={rec['epoch']} loss={rec.get('loss'):.4e}")

cb_count = 0
s.set_update_callback(cb)

print('Starting train, 200 epochs...')
hist = s.train(
    ic_func=ic_func,
    bc_funcs=(bc_left, bc_right),
    N_f=2000,
    N_i=100,
    N_b=100,
    epochs=200,
    learning_rate=1e-3,
    verbose=True,
    log_interval=25,
)
print(f'Train done in {time.time()-t0:.1f}s, {len(hist)} records')
print('Final loss:', hist[-1]['loss'])

print('Predict grid w/ MC=10...')
res = s.predict_grid(nx=50, nt=50, mc_samples=10)
print(f'predict returned {len(res)} values')
if len(res) == 5:
    x, t, Umean, Ustd, Uunc = res
    uflat = Uunc.numpy().flatten() if hasattr(Uunc, 'numpy') else np.array(Uunc).flatten()
    print(f'  has_unc=True, Uunc shape={Uunc.shape if hasattr(Uunc, "shape") else "?"}, max={uflat.max():.3e} mean={uflat.mean():.3e}')
else:
    x, t, U = res[:3]
    print(f'  has_unc=False, U shape check')

print('ALL OK')
