from typing import List

import pandas as pd

from src.common.func import Func
from src.common.point import Point2D
from src.tasks.task3_1_reverse_interpolation.preparation import evenly_divide_into_nodes

def df_from_points(points: List[Point2D]):
    df = pd.DataFrame(columns=('x', 'y'))
    df['x'] = [point.x for point in points]
    df['y'] = [point.y for point in points]
    return df


def build_nodes_df(func: Func, xs):
    df = pd.DataFrame(columns=('x', 'y'))
    df['x'] = xs
    df['y'] = df.x.apply(lambda x: func.f(x))
    return df


def pick_closest(points: List[Point2D], x: float, num: int):
    df = df_from_points(points)
    df['distance'] = df.x.apply(lambda this_x: abs(this_x - x)).sort_values(by='distance').reset_index(drop=True)
    selected_nodes = df[['x', 'y']].iloc[:num].apply(lambda row: Point2D(row[0], row[1]), axis=1).to_list()
    return (df, selected_nodes)
