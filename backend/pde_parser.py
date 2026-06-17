import numpy as np
import sympy as sp
from sympy.parsing.latex import parse_latex


SYMPY_TO_NUMPY = {
    sp.sin: np.sin, sp.cos: np.cos, sp.tan: np.tan,
    sp.exp: np.exp, sp.log: np.log, sp.sqrt: np.sqrt,
    sp.Abs: np.abs, sp.sinh: np.sinh, sp.cosh: np.cosh,
    sp.tanh: np.tanh, sp.pi: np.pi,
}


class PDEParser:
    def __init__(self):
        self.x_sym = sp.Symbol('x')
        self.y_sym = sp.Symbol('y')
        self.t_sym = sp.Symbol('t')
        self.u_sym = sp.Function('u')
        self.variables = {'x': self.x_sym, 'y': self.y_sym, 't': self.t_sym}

    def parse_equation(self, latex_str):
        lhs, rhs = self._split_equation(latex_str)
        lhs_expr = self._parse_side(lhs)
        rhs_expr = self._parse_side(rhs)
        residual = lhs_expr - rhs_expr
        residual = self._replace_derivatives(residual)
        return self._classify_pde(residual), residual

    def _split_equation(self, latex_str):
        latex_str = latex_str.strip()
        if '=' in latex_str:
            parts = latex_str.split('=', 1)
            return parts[0].strip(), parts[1].strip()
        return latex_str, '0'

    def _parse_side(self, latex_str):
        try:
            expr = parse_latex(latex_str)
        except Exception:
            expr = sp.sympify(latex_str, locals={'u': self.u_sym(self.x_sym, self.t_sym)})
        return expr

    def _replace_derivatives(self, expr):
        expr = expr.replace(
            sp.Derivative,
            lambda *args: sp.Symbol(self._derivative_name(args[0], args[1:]))
        )
        u_func = self.u_sym(self.x_sym, self.t_sym)
        expr = expr.subs(u_func, sp.Symbol('u'))
        return expr

    def _derivative_name(self, func, orders):
        name = 'u'
        for var, order in orders:
            for _ in range(order):
                name += f'_{str(var)}'
        return name

    def _classify_pde(self, residual):
        residual_str = str(residual)
        if 'u_t_t' in residual_str or 'u_tt' in residual_str:
            return 'wave'
        elif 'u_t' in residual_str:
            return 'heat'
        elif 'u_x_x' in residual_str and 'u_t' not in residual_str:
            return 'elliptic'
        return 'general'

    def residual_to_lambda(self, residual):
        symbols = sorted(residual.free_symbols, key=lambda s: str(s))
        param_names = [str(s) for s in symbols]
        f = sp.lambdify(symbols, residual, modules=['numpy'])
        return f, param_names

    def parse_initial_condition(self, ic_str):
        try:
            expr = parse_latex(ic_str)
        except Exception:
            expr = sp.sympify(ic_str, locals={'u': sp.Symbol('u')})
        symbols = sorted(expr.free_symbols, key=lambda s: str(s))
        param_names = [str(s) for s in symbols]
        f = sp.lambdify(symbols, expr, modules=['numpy'])
        return f, param_names, str(expr)

    def parse_boundary_condition(self, bc_str):
        return self.parse_initial_condition(bc_str)
