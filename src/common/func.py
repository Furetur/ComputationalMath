from dataclasses import dataclass
from typing import Callable

from src.common.segment import Segment

RealFunction = Callable[[float], float]

@dataclass
class Func:
    f: RealFunction
    df: RealFunction
    domain: Segment

    def has_different_signs_on_ends(self, segment: Segment):
        return self.f(segment.start) * self.f(segment.end) <= 0