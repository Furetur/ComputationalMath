from typing import List

from src.common.point import Point2D
from src.common.utils import product
from src.tasks.task2_interpolation.interpolators.interpolator import Interpolator


class LagrangianInterpolator(Interpolator):
    method_name = 'Lagrangian Interpolator'

    def calc_approximate_value(self, x: float, nodes: List[Point2D]) -> float:
        for node in nodes:
            if node.x == x:
                return node.y

        return sum(
            self.__coef(x, k, nodes) * nodes[k].y
            for k in range(len(nodes))
        )

    @staticmethod
    def __coef(x: float, k: int, nodes: List[Point2D]) -> float:
        indices_except_k = filter(lambda i: i != k, range(len(nodes)))
        x_k = nodes[k].x
        return product(
            ((x - nodes[i].x) / (x_k - nodes[i].x) for i in indices_except_k)
        )
