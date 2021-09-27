from abc import ABC, abstractmethod
from typing import Optional

from src.common.func import Func
from src.common.segment import Segment
from src.tasks.task1_find_roots.root_approximators.root_approximator import RootApproximator


class AbstractNewtonApproximator(RootApproximator, ABC):
    def __init__(self, n_iterations: float):
        self.n_iterations = n_iterations

    @abstractmethod
    def calc_next_approximation(self, func: Func, cur_approximation: float, initial_approximation: float) -> float:
        raise NotImplementedError

    def approximate(
            self,
            func: Func,
            root_domain: Segment,
            initial_approximation: float,
            epsilon: float
    ) -> Optional[float]:
        cur_approx = initial_approximation
        next_approx = self.calc_next_approximation(func, cur_approx, initial_approximation)

        cur_iteration_n = 1
        while abs(cur_approx - next_approx) >= epsilon:
            if cur_iteration_n >= self.n_iterations:
                return None
            cur_approx = next_approx
            next_approx = self.calc_next_approximation(func, cur_approx, initial_approximation)
            cur_iteration_n += 1

        print(f"\t\tn_steps {cur_iteration_n}")
        print(f"\t\tLast |x_m - x_(m-1)| = {abs(cur_approx - next_approx)}")
        return next_approx


class NewtonApproximator(AbstractNewtonApproximator):
    method_name = "Newton"

    def __init__(self, n_iterations: int):
        super().__init__(n_iterations)

    def calc_next_approximation(self, func: Func, cur_approximation: float, initial_approximation: float) -> float:
        return cur_approximation - func.f(cur_approximation) / func.df(cur_approximation)


class ModifiedNewtonApproximator(AbstractNewtonApproximator):
    method_name = "Modified Newton"

    def __init__(self, n_iterations: int):
        super().__init__(n_iterations)

    def calc_next_approximation(self, func: Func, cur_approximation: float, initial_approximation: float) -> float:
        return cur_approximation - func.f(cur_approximation) / func.df(initial_approximation)
