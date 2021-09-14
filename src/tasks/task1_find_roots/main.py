from typing import List

from src.common.func import Func
from src.common.segment import Segment
import yaml
from sympy import lambdify, parse_expr, diff

from src.tasks.task1_find_roots.root_approximators.bisection_approximator import BisectionApproximator
from src.tasks.task1_find_roots.root_approximators.newton_approximator import NewtonApproximator, \
    ModifiedNewtonApproximator
from src.tasks.task1_find_roots.root_approximators.secant_approximator import SecantApproximator


def main():
    with open("input/task1.yaml", "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)

        f = parse_expr(config["f"])
        df = diff(f)
        domain = Segment(int(config["left"]), int(config["right"]))
        func = Func(lambdify("x", f), lambdify("x", df), domain)
        N = int(config["N"])
        n_iterations = int(config["n_iterations"])
        epsilon = float(config["epsilon"])

        print("=== Finding Roots ===")
        print(f"f(x) = {f}")
        print(f"f'(x) = {df}")
        print(f"Domain = {domain}")
        print(f"N = {N}")
        print(f"Epsilon = {epsilon}")
        print(f"n_iterations = {n_iterations}")

        print("1. Isolating roots")

        partitions = isolate_roots(func, N)
        print(f"\t{len(partitions)} partitions:", *partitions)

        print("2. Approximating roots")

        root_approximators = [BisectionApproximator(), NewtonApproximator(n_iterations),
                              ModifiedNewtonApproximator(n_iterations), SecantApproximator(n_iterations)]

        for i, approximator in enumerate(root_approximators):
            print(f"2.{i + 1}. {approximator.method_name}")
            for partition in partitions:
                print(f"\tFor root in {partition}")
                print(f"\t\tInitial approximation: {partition.end}")
                try:
                    result = approximator.approximate(func, partition, partition.end, epsilon)
                except Exception as e:
                    print(f"\t\tApproximator FAILED: {e}")
                print(f"\t\tResult: {result}")
                print(f"\t\tError: {abs(func.f(result))}")


def isolate_roots(func: Func, n_partitions: int) -> List[Segment]:
    result = []
    for partition in func.domain.split(n_partitions):
        if func.has_different_signs_on_ends(partition):
            result.append(partition)
    return result


if __name__ == '__main__':
    main()
