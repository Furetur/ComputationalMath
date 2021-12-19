from typing import Iterable

from scipy.optimize import minimize_scalar
import sympy as sp
from sympy import Symbol


def product(values: Iterable[float]):
    result = 1
    for value in values:
        result *= value
    return result


def minimize(f, a, b):
    res = minimize_scalar(f, bounds=(a, b), method='bounded')
    if not res.success:
        raise Exception(f"Failed to minimize function {res}")
    return res.x, f(res.x)


def maximize(f, a, b):
    x, y = minimize(lambda x: -f(x), a, b)
    return x, -y


def integrate(f_expr, segment) -> float:
    return float(sp.integrate(f_expr, (Symbol("x"), segment.start, segment.end)))


def integrate2(f_expr, segment) -> float:
    F_expr = f_expr.integrate()
    F_lambda = sp.lambdify("x", F_expr)
    return F_lambda(segment.end) - F_lambda(segment.start)


def err(actual, approx):
    return abs(actual - approx)


def rel_err(actual, approx):
    return err(actual, approx) / abs(actual) * 100


def np_poly_to_sympy_expr(np_poly):
    return sp.Poly(reversed(np_poly.coef), Symbol("x")).as_expr()
