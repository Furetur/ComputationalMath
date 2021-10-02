from random import uniform
from typing import List

import streamlit as st


@st.cache
def randomly_select_nodes(a: float, b: float, num: int) -> List[float]:
    points = set()
    while len(points) < num:
        points.add(uniform(a, b))
    return list(sorted(points))


