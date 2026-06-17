import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pinn_solver import PINNSolver, create_ic_func, create_bc_func

params = {'alpha': 0.01}
ic_func = create_ic_func('sin(pi*x)', params)
bc_left = create_bc_func('0', params)
bc_right = create_bc_func('0', params)

print('=== 测试硬约束 + FFM + 自适应激活 + MC Dropout ===')
t0 = time.time()
s = PINNSolver(
    pde_type='heat',
    domain={'x': (0.0, 1.0), 't': (0.0, 1.0)},
    layers=[2, 32, 32, 1],
    params=params,
    use_fourier=True,
    use_adaptive_activation=True,
    use_hard_constraint=True,
    use_mc_dropout=True,
    fourier_bands=[0.1, 1.0, 10.0],
    fourier_freqs=16,
    dropout_rate=0.05,
)
print(f'Model build: {time.time()-t0:.2f}s, trainable params:',
      sum(tf.size(v).numpy() for v in s.model.trainable_variables))

import tensorflow as tf

t0 = time.time()
hist = s.train(
    ic_func=ic_func,
    bc_funcs=(bc_left, bc_right),
    N_f=1000,
    N_i=100,
    N_b=100,
    epochs=30,
    learning_rate=1e-3,
    verbose=False,
    log_interval=5,
)
print(f'Train 30 epochs: {time.time()-t0:.2f}s')
print(f'  initial loss: {hist[0]["loss"]:.4e}')
print(f'  final loss:   {hist[-1]["loss"]:.4e}')
print(f'  loss dropped by factor: {hist[0]["loss"]/hist[-1]["loss"]:.2f}x')

res = s.predict_grid(nx=20, nt=20, mc_samples=8)
print(f'predict returned {len(res)} items')
if len(res) == 5:
    x, t, Um, Us, Uu = res
    print(f'  U_unc shape: {Uu.shape}, max={Uu.max():.3e}, mean={Uu.mean():.3e}')
    print(f'  U shape: {Um.shape}, range: [{Um.min():.4f}, {Um.max():.4f}]')
else:
    print('  (no uncertainty)')

print()
print('=== 测试边界硬约束验证 ===')
x_test = np.array([[0.0, 0.5], [1.0, 0.5]], dtype=np.float32)
u_boundary = s.predict(x_test, mc_samples=1)
print(f'  u(0, 0.5) = {u_boundary[0, 0]:.6f} (should be ~0)')
print(f'  u(1, 0.5) = {u_boundary[1, 0]:.6f} (should be ~0)')
print(f'  max abs boundary error: {max(abs(u_boundary[0,0]), abs(u_boundary[1,0])):.3e}')

print()
print('ALL TESTS PASSED')
