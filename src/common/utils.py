from typing import Iterable

from scipy.optimize import minimize_scalar


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
