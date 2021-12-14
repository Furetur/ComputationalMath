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

from src.common.qf.qf_utils import calc_qf, remap_qf, plot_qf
from src.common.polynomial.legendre import legendres
from src.common.qf.gauss import gauss_qf
from src.common.segment import Segment
from src.common.streamlit import function_input, segment_input

st.write("# 5.2 Вычисление интеграла с помощью КФ Гаусса")


with st.form('main'):
    f_expr, f_lambda = function_input("sqrt(x)*sin(x**2)")
    a, b = segment_input(default_a=0.0, default_b=1.0)
    st.form_submit_button()

st.sidebar.write("## Дополнительные параметры")
N = st.sidebar.number_input(label='N', value=5, min_value=1)

orig_nodes, orig_coefs = gauss_qf(N)
nodes, coefs = remap_qf(orig_nodes, orig_coefs, from_segment=Segment(-1, 1), to_segment=Segment(a, b))

st.write("## Узлы и коэф. КФ")
# plot both qfs
fig, axs = plt.subplots(1, 2)

plot_qf(orig_nodes, orig_coefs, axs[0])
axs[0].set_xlim([-1, 1])
axs[0].set_title('КФ Гаусса')

plot_qf(nodes, coefs, axs[1])
axs[1].set_xlim([a, b])
axs[1].set_title('Подобная КФ')

st.pyplot(fig)

with st.expander(label='Узлы подобной КФ'):
    gauss_df = pd.DataFrame({"Узлы": nodes, "Коэф.": coefs})
    st.dataframe(gauss_df)

st.write("## Вычисление интеграла")

J = float(sp.integrate(f_expr, (x_symbol, a, b)))
J_approx = calc_qf(f_lambda, nodes, coefs)

st.latex(fr"J = \int_{{{a}}}^{{{b}}} f = {J}")
st.latex(fr"J_{{approx}} = {J_approx}")
st.latex(fr"Error = {abs(J - J_approx)}")

