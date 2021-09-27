from typing import Optional

from src.common.func import Func
from src.common.segment import Segment

from src.tasks.task1_find_roots.root_approximators.root_approximator import RootApproximator


class SecantApproximator(RootApproximator):
    method_name = "Secant"

    def __init__(self, n_iterations: float):
        self.n_iterations = n_iterations

    def calc_next_approximation(self, func: Func, cur_x: float, prev_x: float) -> float:
        return cur_x - func.f(cur_x) * (cur_x - prev_x) / (func.f(cur_x) - func.f(prev_x))

    def approximate(
            self,
            func: Func,
            root_domain: Segment,
            initial_approximation: float,
            epsilon: float
    ) -> Optional[float]:
        prev_approx = root_domain.mid()
        cur_approx = initial_approximation
        next_approx = self.calc_next_approximation(func, cur_approx, prev_approx)

        cur_iteration_n = 1
        while abs(cur_approx - next_approx) >= epsilon:
            if cur_iteration_n >= self.n_iterations:
                return None
            prev_approx = cur_approx
            cur_approx = next_approx
            next_approx = self.calc_next_approximation(func, cur_approx, prev_approx)
            cur_iteration_n += 1

        print(f"\t\tn_steps {cur_iteration_n}")
        print(f"\t\tLast |x_m - x_(m-1)| = {abs(cur_approx - next_approx)}")
        return next_approx
