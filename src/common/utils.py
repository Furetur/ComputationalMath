from typing import Iterable


def product(values: Iterable[float]):
    result = 1
    for value in values:
        result *= value
    return result
