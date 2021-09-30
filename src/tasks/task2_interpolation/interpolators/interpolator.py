from typing import Protocol, List, Tuple

from src.common.point import Point2D


class Interpolator(Protocol):
    method_name: str

    def calc_approximate_value(self, x: float, nodes: List[Point2D]) -> float: ...
