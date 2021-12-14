from math import factorial

import numpy as np

from src.common.func import Func
from src.common.polynomial.legendre import legendres
from src.common.processes.find_roots import find_roots
from src.common.segment import Segment
from src.tasks.task1_find_roots.main import isolate_roots
from src.tasks.task1_find_roots.root_approximators.secant_approximator import SecantApproximator


def gauss_qf(n: int):
    P = legendres(n)
    domain = Segment(-1, 1)
    P_n_func = Func(f=P[n], df=None, domain=domain)
    # find roots
    approx = SecantApproximator(max_n_iterations=1000)
    parts = isolate_roots(P_n_func, n_partitions=1000)
    roots = []
    for part in parts:
        root = approx.approximate(P_n_func, part, part.start, epsilon=10 ** -12)
        roots.append(root)
    # nodes
    nodes = roots
    # coefs
    def coef(x_k):
        return (2 * (1 - x_k ** 2)) / (n ** 2 * P[n - 1](x_k) ** 2)
    coefs = [coef(x_k) for x_k in nodes]
    # done
    return nodes, coefs