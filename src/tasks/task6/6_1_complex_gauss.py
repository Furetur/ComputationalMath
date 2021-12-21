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

from src.common.qf.complex_gauss import calc_complex_gauss
from src.common.utils import integrate, err, rel_err
from src.common.qf.qf_utils import calc_qf, plot_qf
from src.common.polynomial.legendre import legendres
from src.common.qf.gauss import gauss_qf
from src.common.segment import Segment
from src.common.streamlit import function_input, segment_input

st.write("# 6.1 Применение составной КФ Гаусса")

with st.form('main'):
    f_expr, f_lambda = function_input("sqrt(1-x)*sin(x)")
    a, b = segment_input(default_a=0.0, default_b=1.0)
    st.form_submit_button()

N = st.sidebar.number_input(min_value=1, value=2 , label='N: кол-во узлов')
m = st.sidebar.number_input(min_value=1, label='m: кол-во разбиений')

domain = Segment(a, b)
partitions = domain.split(m)

st.write(f_expr)

J = integrate(f_expr, domain)
J_approx = calc_complex_gauss(N, partitions, f_lambda)

with st.expander(label='Более подробно'):
    nodes, coefs = gauss_qf(N)
    gauss_df = pd.DataFrame({"Узлы": nodes, "Коэф.": coefs})
    st.dataframe(gauss_df)

st.latex(f"J = {J}")
st.latex(f"J_{{approx}} = {J_approx}")
st.latex(f"Error = {err(J, J_approx)}")
st.latex(f"Rel.Error = {rel_err(J, J_approx)}\%")

