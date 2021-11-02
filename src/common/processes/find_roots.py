from typing import List, Optional

from src.common.func import Func
from src.common.segment import Segment
from src.tasks.task1_find_roots.main import isolate_roots
from src.tasks.task1_find_roots.root_approximators.bisection_approximator import BisectionApproximator


def find_roots(func: Func, epsilon: float, domain: Optional[Segment] = None) -> List[float]:
    partitions = isolate_roots(func, n_partitions=1000, domain=domain)
    approximator = BisectionApproximator()

    roots = []

    for partition in partitions:
        root = approximator.approximate(func, partition, partition.start, epsilon)
        roots.append(root)

    return roots
