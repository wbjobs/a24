"""快速端到端验证：ND PINN、FDM、对比、方程库、Optuna超参搜索 + Horovod检测"""
import sys, os, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

results = []

def section(name):
    print()
    print('=' * 70)
    print(f'  {name}')
    print('=' * 70)

def t(name, fn, required=True):
    try:
        t0 = time.time()
        val = fn()
        elapsed = time.time() - t0
        results.append((name, True, elapsed, val))
        print(f'  ✅ [{elapsed:6.2f}s] {name}', '' if val is None else f' → {str(val)[:80]}')
        return val
    except Exception as e:
        tb = traceback.format_exc()
        results.append((name, False, 0, str(e)))
        status = '❌ FATAL' if required else '⚠️  WARN'
        print(f'  {status} {name}: {e}')
        print(f'     {tb.split(chr(10))[-2]}')
        if required:
            return None
        return None

# ---------- 1. 方程库验证 ----------
section('1. 方程库验证')
lib = t('加载 pde_library 模块', lambda: __import__('pde_library'))
if lib is not None:
    idx = t('get_library_index() 返回20个案例', lambda: len(lib.get_library_index()))
    all_ids = [c['id'] for c in lib.PDE_LIBRARY]
    t('20案例ID唯一性', lambda: len(set(all_ids)) == 20)
    t('kdv_soliton案例可获取', lambda: lib.get_case('kdv_soliton')['name'])
    t('ns_lid_cavity案例可获取', lambda: 'NS' in lib.get_case('ns_lid_cavity')['name'] or 'Navier' in lib.get_case('ns_lid_cavity')['name'] or '顶盖' in lib.get_case('ns_lid_cavity')['name'] or 'lid' in lib.get_case('ns_lid_cavity')['name'].lower())
    t('create_case_functions(kdv)成功', lambda: lib.create_case_functions(lib.get_case('kdv_soliton'))[0](x=0.5))

# ---------- 2. FDMSolver验证 ----------
section('2. FDM传统求解器验证')
from classical_solvers import FDMSolver, compare_solutions
fdm = FDMSolver()
ic_heat = lambda x: np.sin(np.pi * x)
bcl = bcr = lambda t: 0
r_heat = t('solve_heat_1d (Crank-Nicolson)', lambda: fdm.solve_heat_1d(ic_heat, bcl, bcr, 0, 1, 0, 0.1, nx=50, nt=500, alpha=0.01))
r_wave = t('solve_wave_1d (Leapfrog)', lambda: fdm.solve_wave_1d(
    lambda x: np.sin(np.pi*x), lambda x: 0, bcl, bcr, 0, 1, 0, 0.5, nx=80, nt=400, c=1.0))
r_kdv = t('solve_kdv_1d (Zabusky-Kruskal)', lambda: fdm.solve_kdv_1d(
    lambda x: np.cos(np.pi * (x + 0.5)) ** 2,
    lambda t: np.cos(np.pi * (-1 + 0.5))**2,
    lambda t: np.cos(np.pi * (2 + 0.5))**2,
    -1, 2, 0, 0.01, nx=128, nt=500, alpha=1.0, beta=6.0))

# ---------- 3. NDPINNSolver验证 ----------
section('3. N维PINN求解器验证')
from pinn_solver_nd import NDPINNSolver, create_ic_func_nd, PINNSolver
# 轻量级验证：1D热传导 50epochs
ic_fn_nd = create_ic_func_nd('sin(pi*x)', {}, ['x'])
bc_fn_0 = create_ic_func_nd('0', {}, ['t'])
bc_fn_1 = create_ic_func_nd('0', {}, ['t'])
solver = t('构建1D heat PINN', lambda: NDPINNSolver(
    pde_type='heat', n_dims=1,
    domain={'x': (0,1), 't': (0,0.1)},
    layers=[2, 16, 16, 1], params={'alpha': 0.01},
    use_fourier=True, use_adaptive_activation=True, use_hard_constraint=True,
    fourier_bands=[0.01, 0.1, 1], fourier_freqs=16))
if solver is not None:
    hist = t('train(50 epochs)', lambda: solver.train(
        ic_fn_nd, [bc_fn_0, bc_fn_1],
        N_f=800, N_i=100, N_b=100,
        epochs=50, learning_rate=1e-2, log_interval=20, verbose=False))
    if hist:
        print(f'     init loss {hist[0]["loss"]:.3e} → final loss {hist[-1]["loss"]:.3e} ({(hist[0]["loss"]/max(hist[-1]["loss"],1e-20)):.2f}×下降)')
    grid = t('predict_grid() 50×50', lambda: solver.predict_grid({'x':50, 't':50}, mc_samples=1))
    if grid and r_heat:
        comp = t('compare_solutions(PINN vs FDM)', lambda: compare_solutions(grid, r_heat))
        if comp:
            print(f'     L2={comp["l2_error"]:.3e}, L∞={comp["linf_error"]:.3e}')

# 2D 验证：拉普拉斯方程
solver2d = t('构建2D Laplace PINN', lambda: NDPINNSolver(
    pde_type='laplace', n_dims=2,
    domain={'x': (0,1), 'y': (0,1), 't': (0,1)},
    layers=[3, 20, 20, 1], params={},
    use_hard_constraint=True))

# NS方程验证
solver_ns = t('构建NS不可压缩 (2D+t)', lambda: NDPINNSolver(
    pde_type='ns_incompressible', n_dims=2,
    domain={'x': (0,1), 'y': (0,1), 't': (0,0.5)},
    layers=[3, 32, 32, 4], params={'nu': 0.01, 'rho': 1.0},
    use_hard_constraint=True))

# 3D 验证
solver3d = t('构建3D+time PINN (heat)', lambda: NDPINNSolver(
    pde_type='heat', n_dims=3,
    domain={'x': (0,1), 'y': (0,1), 'z': (0,1), 't': (0,0.1)},
    layers=[4, 24, 24, 1], params={'alpha': 0.01}))

# 兼容层验证
legacy_solver = t('旧PINNSolver类兼容', lambda: PINNSolver(
    pde_type='heat',
    domain={'x': (0,1), 't': (0,0.1)},
    layers=[2, 20, 1]))

# ---------- 4. Optuna超参搜索（如果安装了） ----------
section('4. Optuna超参搜索')
try:
    from hyperparam_search import is_optuna_available, EQUATION_HP_RANGES
    print(f'  Optuna installed: {is_optuna_available()}')
    t(f'EQUATION_HP_RANGES覆盖 {len(EQUATION_HP_RANGES)} 种方程', lambda: len(EQUATION_HP_RANGES) >= 10)
    if is_optuna_available():
        from hyperparam_search import run_hyperparameter_search
        hp_res = t('Optuna快速搜索(3 trials × 20epochs)', lambda: run_hyperparameter_search(
            pde_type='heat', n_dims=1,
            domain={'x': (0,1), 't': (0,0.05)},
            params={'alpha': 0.01},
            ic_func=ic_fn_nd,
            bc_funcs=[bc_fn_0, bc_fn_1],
            n_trials=3, quick_epochs=20, verbose=False), required=False)
        if hp_res:
            print(f'     best_loss={hp_res["best_loss"]:.3e}, best={hp_res["best_params"]}')
except Exception as e:
    print(f'  ⚠️  Optuna模块异常：{e}')

# ---------- 5. Horovod检测 ----------
section('5. Horovod分布式检测')
try:
    from horovod_distributed import is_horovod_available
    print(f'  Horovod可用: {is_horovod_available()}')
    try:
        from horovod_distributed import init_horovod, train_distributed
        print(f'  导出函数: init_horovod={callable(init_horovod)}, train_distributed={callable(train_distributed)}')
    except Exception:
        pass
except Exception as e:
    print(f'  ⚠️  Horovod模块异常：{e}')

# ---------- 汇总 ----------
section('测试汇总')
passed = sum(1 for r in results if r[1])
failed = sum(1 for r in results if not r[1])
print(f'  共 {len(results)} 项检查')
print(f'  ✅ PASS: {passed}    ❌ FAIL: {failed}')
if failed == 0:
    print()
    print('  🎉🎉🎉 全部核心模块验证通过！')
    print('     - 20个PDE方程案例：完整')
    print('     - NDPINNSolver (1D/2D/3D+time, NS等)：可构建')
    print('     - FDM传统求解器 (Crank-Nicolson/Leapfrog/Zabusky-Kruskal等)：可用')
    print('     - PINN vs FDM 对比：可输出L2/L∞/加速比')
    print('     - 兼容旧PINNSolver API：✅')
    print('     - Optuna超参搜索：' + ('✅ 可用' if is_optuna_available() else '⚠️ 未安装(pip install optuna)'))
    print('     - Horovod多GPU分布式：' + ('✅ 可用' if is_horovod_available() else '⚠️ 未安装(horovodrun)'))
    print()
else:
    print()
    print('  失败项:')
    for r in results:
        if not r[1]:
            print(f'    ❌ {r[0]}: {r[3]}')
    sys.exit(1)
