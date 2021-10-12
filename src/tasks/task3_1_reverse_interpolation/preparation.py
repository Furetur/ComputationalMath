from random import uniform
from typing import List

import streamlit as st


@st.cache
def evenly_divide_into_nodes(a: float, b: float, num: int) -> List[float]:
    h = (b - a) / (num - 1)
    return [a + h * i for i in range(num)]
