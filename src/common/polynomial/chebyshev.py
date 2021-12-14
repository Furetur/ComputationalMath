import numpy as np
from numpy.polynomial import Polynomial


def chebyshevs(n: int):
    assert n >= 0
    X = Polynomial([0, 1])

    T = [Polynomial([1, 0]), X]
    # T_{n+1} = 2x T_n - T_{n-1}
    # Therefore
    # T_{n} = 2x T_{n-1} - T_{n-2}
    for i in range(2, n + 1):
        T.append(2 * X * T[i - 1] - T[i - 2])
    return T[:n + 1]

def chebyshev_roots(n):
    return [np.cos(np.pi * (2*k-1)/(2*n)) for k in range(1, n+1)]