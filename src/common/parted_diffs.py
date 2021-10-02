from typing import List, Tuple
import numpy as np

from src.common.point import Point2D


class PartedDiffsCalculator:
    def get_parted_diff(self, points: Tuple[Point2D]) -> float:
        if len(points) == 0:
            raise Exception("points was empty")
        if type(points) == list:
            points = tuple(points)
        elif len(points) == 1:
            return points[0].y
        else:
            first, *tail = points
            *head, last = points
            return (self.get_parted_diff(tuple(tail)) - self.get_parted_diff(tuple(head))) / (last.x - first.x)


def calc_parted_diffs(points: List[Point2D]) -> List[float]:
    """
    :param points: (x_0, y_0) ... (x_n, y_n)
    :return: diffs where diffs[i] = f(x_0...x_i) parted diff
    """
    if len(points) == 0:
        raise Exception("points was empty")
    n = len(points)
    # table[i][j] = parted diff f(x_{i}...x_{i+j})
    table = np.zeros([n, n])
    # set the first column to y_0 ... y_n.
    # In other words table[i][0] = f(x_i..x_i) = f(x_i) = y_i
    for i in range(n):
        table[i, 0] = points[i].y

    for j in range(1,n):
        for i in range(n-j):
            table[i, j] = (table[i+1, j-1] - table[i, j-1]) / (points[i+j].x-points[i].x)

    return table[0, :]

