from typing import List

from src.common.parted_diffs import PartedDiffsCalculator, calc_parted_diffs
from src.common.point import Point2D
from src.tasks.task2_interpolation.interpolators.interpolator import Interpolator


class NewtonInterpolator(Interpolator):
    method_name = 'Newton Interpolator'

    def __init__(self):
        self.parted_diff_calculator = PartedDiffsCalculator()

    def calc_approximate_value(self, x: float, nodes: List[Point2D]) -> float:
        coefs = calc_parted_diffs(nodes)
        cur_multiple = 1
        total_value = coefs[0]
        for i in range(1, len(nodes)):
            cur_multiple *= x - nodes[i - 1].x
            total_value += coefs[i] * cur_multiple
        return total_value

    def __calc_coefs(self, nodes: List[Point2D]) -> List[float]:
        coefs = []
        for i in range(1, len(nodes) + 1):
            cur_args = tuple(nodes[:i])
            coefs.append(self.parted_diff_calculator.get_parted_diff(cur_args))
        return coefs
