"""传统数值方法（有限差分FDM）求解器 - 用于和PINN对比误差/速度"""
import time
import numpy as np


class FDMSolver:
    """经典有限差分求解器，支持热传导、波动、Burgers、Allen-Cahn、KdV等"""

    def solve_heat_1d(self, ic_func, bc_left, bc_right,
                      x_min=0, x_max=1, t_min=0, t_max=1,
                      nx=200, nt=10000, alpha=0.01):
        """1D热传导方程，Crank-Nicolson格式"""
        t0 = time.time()
        dx = (x_max - x_min) / (nx - 1)
        dt = (t_max - t_min) / (nt - 1)
        r = alpha * dt / (dx ** 2)

        x = np.linspace(x_min, x_max, nx).astype(np.float32)
        t = np.linspace(t_min, t_max, nt).astype(np.float32)

        U = np.zeros((nt, nx), dtype=np.float32)
        U[0, :] = ic_func(x).astype(np.float32).reshape(-1)

        A = np.diag((1 + r) * np.ones(nx - 2)) + \
            np.diag(-r / 2 * np.ones(nx - 3), 1) + \
            np.diag(-r / 2 * np.ones(nx - 3), -1)

        B = np.diag((1 - r) * np.ones(nx - 2)) + \
            np.diag(r / 2 * np.ones(nx - 3), 1) + \
            np.diag(r / 2 * np.ones(nx - 3), -1)

        for n in range(nt - 1):
            tn = t[n]
            tn1 = t[n + 1]
            rhs = B @ U[n, 1:-1]
            rhs[0] += r / 2 * (bc_left(tn) + bc_left(tn1))
            rhs[-1] += r / 2 * (bc_right(tn) + bc_right(tn1))
            U[n + 1, 1:-1] = np.linalg.solve(A, rhs)
            U[n + 1, 0] = bc_left(tn1)
            U[n + 1, -1] = bc_right(tn1)

        elapsed = time.time() - t0
        return {
            'x': x, 't': t, 'u': U,
            'nx': nx, 'nt': nt, 'dx': dx, 'dt': dt,
            'elapsed_sec': elapsed,
            'method': 'Crank-Nicolson FDM',
        }

    def solve_wave_1d(self, ic_func, ic_dt_func, bc_left, bc_right,
                      x_min=0, x_max=1, t_min=0, t_max=1,
                      nx=200, nt=2000, c=1.0):
        """1D波动方程，蛙跳格式"""
        t0 = time.time()
        dx = (x_max - x_min) / (nx - 1)
        dt = (t_max - t_min) / (nt - 1)
        r = c * dt / dx
        assert r < 1.0, f'CFL不满足: r={r:.3f} > 1'

        x = np.linspace(x_min, x_max, nx).astype(np.float32)
        t = np.linspace(t_min, t_max, nt).astype(np.float32)

        U = np.zeros((nt, nx), dtype=np.float32)
        U[0, :] = np.broadcast_to(np.asarray(ic_func(x), dtype=np.float32).reshape(-1), (nx,)).copy()

        dudt0_raw = np.asarray(ic_dt_func(x), dtype=np.float32).reshape(-1)
        dudt0 = np.broadcast_to(dudt0_raw, (nx,)).copy()
        U[1, 1:-1] = U[0, 1:-1] + dt * dudt0[1:-1] + 0.5 * r ** 2 * (
            U[0, 2:] - 2 * U[0, 1:-1] + U[0, :-2]
        )
        U[1, 0] = float(np.asarray(bc_left(t[1]), dtype=np.float32).item() if hasattr(np.asarray(bc_left(t[1])), 'item') else bc_left(t[1]))
        U[1, -1] = float(np.asarray(bc_right(t[1]), dtype=np.float32).item() if hasattr(np.asarray(bc_right(t[1])), 'item') else bc_right(t[1]))

        for n in range(1, nt - 1):
            U[n + 1, 1:-1] = 2 * U[n, 1:-1] - U[n - 1, 1:-1] + \
                r ** 2 * (U[n, 2:] - 2 * U[n, 1:-1] + U[n, :-2])
            U[n + 1, 0] = float(np.asarray(bc_left(t[n + 1]), dtype=np.float32).item())
            U[n + 1, -1] = float(np.asarray(bc_right(t[n + 1]), dtype=np.float32).item())

        return {
            'x': x, 't': t, 'u': U,
            'nx': nx, 'nt': nt, 'dx': dx, 'dt': dt,
            'elapsed_sec': time.time() - t0,
            'method': 'Leapfrog FDM',
        }

    def solve_burgers_1d(self, ic_func, bc_left, bc_right,
                         x_min=0, x_max=1, t_min=0, t_max=1,
                         nx=400, nt=20000, nu=0.01):
        """1D Burgers方程，MacCormack格式"""
        t0 = time.time()
        dx = (x_max - x_min) / (nx - 1)
        dt = (t_max - t_min) / (nt - 1)

        x = np.linspace(x_min, x_max, nx).astype(np.float32)
        t = np.linspace(t_min, t_max, nt).astype(np.float32)

        U = np.zeros((nt, nx), dtype=np.float32)
        U[0, :] = ic_func(x).astype(np.float32).reshape(-1)

        for n in range(nt - 1):
            u = U[n, :]
            dudx = np.zeros_like(u)
            dudx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
            d2udx2 = np.zeros_like(u)
            d2udx2[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx ** 2)

            up = u - dt * u * dudx + dt * nu * d2udx2
            up[0] = bc_left(t[n + 1])
            up[-1] = bc_right(t[n + 1])

            dudx_p = np.zeros_like(up)
            dudx_p[1:-1] = (up[2:] - up[:-2]) / (2 * dx)
            d2udx2_p = np.zeros_like(up)
            d2udx2_p[1:-1] = (up[2:] - 2 * up[1:-1] + up[:-2]) / (dx ** 2)

            U[n + 1, 1:-1] = 0.5 * (u[1:-1] + up[1:-1]) - \
                0.5 * dt * u[1:-1] * dudx_p[1:-1] + \
                0.5 * dt * nu * d2udx2_p[1:-1]
            U[n + 1, 0] = bc_left(t[n + 1])
            U[n + 1, -1] = bc_right(t[n + 1])

        return {
            'x': x, 't': t, 'u': U,
            'nx': nx, 'nt': nt, 'dx': dx, 'dt': dt,
            'elapsed_sec': time.time() - t0,
            'method': 'MacCormack FDM',
        }

    def solve_allen_cahn_1d(self, ic_func, bc_left, bc_right,
                            x_min=0, x_max=1, t_min=0, t_max=1,
                            nx=400, nt=10000, epsilon=0.01, gamma=1.0):
        """1D Allen-Cahn方程，半隐式FDM"""
        t0 = time.time()
        dx = (x_max - x_min) / (nx - 1)
        dt = (t_max - t_min) / (nt - 1)

        x = np.linspace(x_min, x_max, nx).astype(np.float32)
        t = np.linspace(t_min, t_max, nt).astype(np.float32)
        U = np.zeros((nt, nx), dtype=np.float32)
        U[0, :] = ic_func(x).astype(np.float32).reshape(-1)

        r = gamma * epsilon ** 2 * dt / (dx ** 2)
        A = np.diag((1 + 2 * r) * np.ones(nx - 2)) + \
            np.diag(-r * np.ones(nx - 3), 1) + \
            np.diag(-r * np.ones(nx - 3), -1)

        for n in range(nt - 1):
            u = U[n, :]
            rhs = u[1:-1] + dt * gamma * u[1:-1] * (1 - u[1:-1] ** 2)
            tn1 = t[n + 1]
            rhs[0] += r * bc_left(tn1)
            rhs[-1] += r * bc_right(tn1)
            U[n + 1, 1:-1] = np.linalg.solve(A, rhs)
            U[n + 1, 0] = bc_left(tn1)
            U[n + 1, -1] = bc_right(tn1)

        return {
            'x': x, 't': t, 'u': U,
            'nx': nx, 'nt': nt, 'dx': dx, 'dt': dt,
            'elapsed_sec': time.time() - t0,
            'method': '半隐式FDM',
        }

    def solve_kdv_1d(self, ic_func, bc_left, bc_right,
                     x_min=0, x_max=1, t_min=0, t_max=1,
                     nx=800, nt=40000, alpha=1.0, beta=6.0):
        """1D KdV方程，Zabusky-Kruskal格式"""
        t0 = time.time()
        dx = (x_max - x_min) / (nx - 1)
        dt = (t_max - t_min) / (nt - 1)

        x = np.linspace(x_min, x_max, nx).astype(np.float32)
        t = np.linspace(t_min, t_max, nt).astype(np.float32)
        U = np.zeros((nt, nx), dtype=np.float32)
        U[0, :] = ic_func(x).astype(np.float32).reshape(-1)

        U[1, :] = U[0, :]

        for n in range(1, nt - 1):
            u = U[n, :]
            um = U[n - 1, :]
            un = np.zeros_like(u)
            for i in range(3, nx - 3):
                dudx = (u[i + 1] - u[i - 1]) / (2 * dx)
                d3udx3 = (u[i + 3] - 3 * u[i + 1] + 3 * u[i - 1] - u[i - 3]) / (8 * dx ** 3)
                un[i] = um[i] - 2 * dt * (alpha * u[i] * dudx + beta * d3udx3)
            for i in range(3):
                un[i] = bc_left(t[n + 1]) if i == 0 else un[i + 1]
            for i in range(nx - 3, nx):
                un[i] = bc_right(t[n + 1]) if i == nx - 1 else un[i - 1]
            U[n + 1, :] = un

        return {
            'x': x, 't': t, 'u': U,
            'nx': nx, 'nt': nt, 'dx': dx, 'dt': dt,
            'elapsed_sec': time.time() - t0,
            'method': 'Zabusky-Kruskal FDM',
        }

    def solve_poisson_2d(self, f_func, bc_left, bc_right, bc_bottom, bc_top,
                         x_min=0, x_max=1, y_min=0, y_max=1,
                         nx=100, ny=100):
        """2D Poisson方程，雅可比迭代"""
        t0 = time.time()
        dx = (x_max - x_min) / (nx - 1)
        dy = (y_max - y_min) / (ny - 1)
        x = np.linspace(x_min, x_max, nx).astype(np.float32)
        y = np.linspace(y_min, y_max, ny).astype(np.float32)
        X, Y = np.meshgrid(x, y)
        f = f_func(X, Y).astype(np.float32)
        U = np.zeros((ny, nx), dtype=np.float32)
        U[0, :] = bc_bottom(x)
        U[-1, :] = bc_top(x)
        U[:, 0] = bc_left(y)
        U[:, -1] = bc_right(y)

        for it in range(20000):
            U_new = U.copy()
            U_new[1:-1, 1:-1] = (
                (U[1:-1, 2:] + U[1:-1, :-2]) / dx ** 2 +
                (U[2:, 1:-1] + U[:-2, 1:-1]) / dy ** 2 -
                f[1:-1, 1:-1]
            ) / (2 / dx ** 2 + 2 / dy ** 2)
            diff = np.linalg.norm(U_new - U)
            U = U_new
            if diff < 1e-6:
                break

        return {
            'x': x, 'y': y, 'u': U,
            'iterations': it + 1,
            'elapsed_sec': time.time() - t0,
            'method': 'Jacobi FDM',
        }

    def solve(self, pde_type, *args, **kwargs):
        """通用入口"""
        dispatch = {
            'heat': self.solve_heat_1d,
            'wave': self.solve_wave_1d,
            'burgers': self.solve_burgers_1d,
            'allen_cahn': self.solve_allen_cahn_1d,
            'kdv': self.solve_kdv_1d,
            'poisson': self.solve_poisson_2d,
        }
        fn = dispatch.get(pde_type)
        if fn is None:
            return {'error': f'FDM does not support {pde_type} yet'}
        return fn(*args, **kwargs)


def compare_solutions(pinn_grid, fdm_result):
    """
    对比PINN与传统方法
    Args:
        pinn_grid: predict_grid兼容的字典，或{x,t,u}
        fdm_result: FDM返回的字典
    Returns:
        对比统计信息
    """
    t0 = time.time()
    fdm_u = np.asarray(fdm_result['u'])
    if 'u_0' in pinn_grid:
        pinn_u = np.asarray(pinn_grid['u_0'])
    elif 'u' in pinn_grid:
        if isinstance(pinn_grid['u'], np.ndarray):
            pinn_u = pinn_grid['u']
        else:
            pinn_u = np.asarray(pinn_grid['u'])
    else:
        return {'error': 'PINN u format unknown'}

    if 'axes' in pinn_grid:
        px = np.asarray(pinn_grid['axes'].get('x', []))
        pt = np.asarray(pinn_grid['axes'].get('t', []))
    else:
        px = np.asarray(pinn_grid.get('x', []))
        pt = np.asarray(pinn_grid.get('t', []))

    fx = np.asarray(fdm_result['x'])
    ft = np.asarray(fdm_result['t'])

    common_nt = min(len(pt), len(ft))
    common_nx = min(len(px), len(fx))

    # 重采样到相同网格
    from scipy.interpolate import RegularGridInterpolator
    try:
        fdm_interp = RegularGridInterpolator((ft, fx), fdm_u, bounds_error=False, fill_value=0)
        Xg, Tg = np.meshgrid(px[:common_nx], pt[:common_nt])
        pts = np.stack([Tg.reshape(-1), Xg.reshape(-1)], axis=-1)
        fdm_rs = fdm_interp(pts).reshape(common_nt, common_nx)
        pinn_rs = pinn_u[:common_nt, :common_nx]

        diff = pinn_rs - fdm_rs
        l2_error = np.sqrt(np.mean(diff ** 2))
        linf_error = np.max(np.abs(diff))
        rmse = np.sqrt(np.mean(diff ** 2))
        rel_error = l2_error / (np.sqrt(np.mean(fdm_rs ** 2)) + 1e-12)
    except Exception as e:
        l2_error = float('nan')
        linf_error = float('nan')
        rmse = float('nan')
        rel_error = float('nan')
        common_nt = 0
        common_nx = 0

    return {
        'pinn_time_sec': pinn_grid.get('training_time_sec', None),
        'fdm_time_sec': fdm_result.get('elapsed_sec', None),
        'speedup_ratio': (fdm_result.get('elapsed_sec', 1) /
                          max(1e-9, pinn_grid.get('training_time_sec', 1e-9))),
        'fdm_method': fdm_result.get('method', ''),
        'l2_error': float(l2_error),
        'linf_error': float(linf_error),
        'rmse': float(rmse),
        'relative_error': float(rel_error),
        'common_nt': common_nt,
        'common_nx': common_nx,
        'elapsed_comparison_sec': time.time() - t0,
    }
