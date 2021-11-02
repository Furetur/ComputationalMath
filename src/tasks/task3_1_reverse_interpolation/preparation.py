from random import uniform
from typing import List

import streamlit as st

from src.common.point import Point2D


@st.cache
def evenly_divide_into_nodes(a: float, b: float, num: int) -> List[float]:
    h = (b - a) / (num - 1)
    return [a + h * i for i in range(num)]


def select_appropriate_points(points: List[Point2D], n: int, y: float) -> List[Point2D]:
    assert len(points) > 0
    sorted_by_y = list(sorted(points, key=lambda p: p.y))
    first_index_where_point_y_greater = first_index_where(sorted_by_y, lambda p: p.y > y)
    if first_index_where_point_y_greater is None:
        left_index = len(points) - 1
        right_index = len(points) - 1
    elif first_index_where_point_y_greater == 0:
        left_index = 0
        right_index = 0
    elif len(points) == 1:
        left_index = 0
        right_index = 0
    else:
        left_index = first_index_where_point_y_greater - 1
        right_index = first_index_where_point_y_greater
    assert 0 <= left_index < len(points) and 0 <= right_index < len(points)
    x_mid = (sorted_by_y[left_index].x + sorted_by_y[right_index].x) / 2
    sorted_by_x_distance = list(sorted(points, key=lambda p: abs(p.x - x_mid)))
    return sorted_by_x_distance[:n]


def first_index_where(arr, predicate):
    for index, el in enumerate(arr):
        if predicate(el):
            return index
