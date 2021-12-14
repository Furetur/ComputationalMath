import sys
from typing import List, Callable, Protocol

import pandas as pd
import streamlit as st
import seaborn as sns
import sympy as sp
from sympy.abc import x as x_symbol
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial


sys.path.append('')
sys.path.append('../../..')

from src.common.qf.qf_utils import calc_qf, plot_qf
from src.common.polynomial.legendre import legendres
from src.common.qf.gauss import gauss_qf
from src.common.segment import Segment
from src.common.streamlit import function_input, segment_input

st.write("# 5.1 Построение КФ Гаусса")

st.sidebar.write("## Параметры задачи")
N = st.sidebar.number_input(label='N', value=5, min_value=1)

st.write("## КФ Гаусса")
# calc gauss
nodes, coefs = gauss_qf(N)

# plot nodes, coefs and legendre
fig, ax = plt.subplots()
plt.xlim([-1, 1])

plot_qf(nodes, coefs, ax)

x, y = legendres(N)[N].linspace(100)
ax.plot(x, y, label='Лежандр')

ax.legend()
st.pyplot(fig)

# display nodes
with st.expander(label='Более подробно'):
    gauss_df = pd.DataFrame({"Узлы": nodes, "Коэф.": coefs})
    st.dataframe(gauss_df)

# verify on polynomial
st.write("## Проверка точности на многочленах")

max_degree = 2 * N - 1
st.write(f"КФ точна для многочленов степени не выше {max_degree}")


def generate_poly():
    global poly
    poly = Polynomial(np.random.randint(low=-50, high=50, size=max_degree + 1))

poly = Polynomial(np.random.randint(low=-50, high=50, size=max_degree + 1))
stpoly_expr = sp.Poly(reversed(poly.coef), x_symbol).as_expr()
st.latex(f"f =")
st.write(stpoly_expr)

st.button(label="Сгенерировать заново", on_click=generate_poly)

a, b = -1, 1
J = float(sp.integrate(stpoly_expr, (x_symbol, a, b)))
J_approx = calc_qf(poly, nodes, coefs)

st.latex(fr"J = \int_{{{a}}}^{{{b}}} f = {J}")
st.latex(fr"J_{{approx}} = {J_approx}")
st.latex(fr"Error = {abs(J - J_approx)}")
