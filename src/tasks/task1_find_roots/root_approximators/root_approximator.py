from typing import Protocol, Optional

from src.common.func import Func
from src.common.segment import Segment


class RootApproximator(Protocol):
    method_name: str

    def approximate(
            self,
            func: Func,
            root_domain: Segment,
            initial_approximation: float,
            epsilon: float
    ) -> Optional[float]: ...
