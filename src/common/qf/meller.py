import numpy as np

from src.common.polynomial.chebyshev import chebyshevs, chebyshev_roots


def meller_qf(n: int):
    return chebyshev_roots(n), [np.pi / n] * n
