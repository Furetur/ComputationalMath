from dataclasses import dataclass
from typing import Callable, List

from sympy import parse_expr, diff, lambdify

from src.common.point import Point2D
from src.common.segment import Segment

RealFunction = Callable[[float], float]

@dataclass
class Func:
    f: RealFunction
    df: RealFunction
    domain: Segment

    @staticmethod
    def from_string(f: str, domain: Segment):
        f_expr = parse_expr(f)
        df = diff(f_expr)
        return Func(lambdify("x", f_expr), lambdify("x", df), domain)

    def has_different_signs_on_ends(self, segment: Segment):
        return self.f(segment.start) * self.f(segment.end) <= 0

    def calculate_points(self, xs: List[float]) -> List[Point2D]:
        return [Point2D(x, self.f(x)) for x in xs]
