from typing import Optional

from src.common.func import Func
from src.common.segment import Segment
from src.tasks.task1_find_roots.root_approximators.root_approximator import RootApproximator


class BisectionApproximator(RootApproximator):
    method_name = "Bisection"

    def approximate(
            self,
            func: Func,
            root_domain: Segment,
            initial_approximation: float,
            epsilon: float
    ) -> Optional[float]:
        n_steps = 1

        cur_segment = root_domain
        while cur_segment.len() >= 2 * epsilon:
            left, right = cur_segment.split(2)
            if func.has_different_signs_on_ends(left):
                cur_segment = left
            else:
                cur_segment = right
            n_steps += 1

        print(f"\t\tn_steps: {n_steps}")
        print(f"\t\tLast length: {cur_segment.len()}")

        return cur_segment.mid()
