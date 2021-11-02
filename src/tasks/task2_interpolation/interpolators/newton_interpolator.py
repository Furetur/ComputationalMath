from typing import List

from src.common.parted_diffs import PartedDiffsCalculator, calc_parted_diffs
from src.common.point import Point2D
from src.tasks.task2_interpolation.interpolators.interpolator import Interpolator


class NewtonInterpolator(Interpolator):
    method_name = 'Newton Interpolator'

    def calc_approximate_value(self, x: float, nodes: List[Point2D]) -> float:
        for node in nodes:
            if node.x == x:
                return node.y

        coefs = calc_parted_diffs(nodes)
        cur_multiple = 1
        total_value = coefs[0]
        for i in range(1, len(nodes)):
            cur_multiple *= x - nodes[i - 1].x
            total_value += coefs[i] * cur_multiple
        return total_value


class ReusableNewtonInterpolator:
    method_name = 'Newton Interpolator'

    def __init__(self, nodes: List[Point2D]):
        assert len(nodes) > 0
        self.nodes = nodes
        self.coefs = calc_parted_diffs(nodes)

    def calc_approximate_value(self, x: float) -> float:
        for node in self.nodes:
            if node.x == x:
                return node.y

        cur_multiple = 1
        total_value = self.coefs[0]
        for i in range(1, len(self.nodes)):
            cur_multiple *= x - self.nodes[i - 1].x
            total_value += self.coefs[i] * cur_multiple
        return total_value
