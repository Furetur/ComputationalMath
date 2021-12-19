import sys
from typing import List, Callable, Protocol

import pandas as pd
import streamlit as st
import seaborn as sns
import sympy as sp
from sympy import Symbol
from sympy.abc import x as x_symbol
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial


sys.path.append('')
sys.path.append('../../..')

from src.common.streamlit_parts.solve_linalg import st_solve_linalg
from src.common.polynomial.omega import omega
from src.common.qf.complex_gauss import calc_complex_gauss
from src.common.utils import integrate, err, rel_err, np_poly_to_sympy_expr, product
from src.common.qf.qf_utils import calc_qf, plot_qf
from src.common.polynomial.legendre import legendres
from src.common.qf.gauss import gauss_qf
from src.common.segment import Segment
from src.common.streamlit import function_input, segment_input

x_symbol = Symbol('x')

st.write("# 6.1 Применение составной КФ Гаусса")

with st.form('main'):
    ro_expr, ro_lambda = function_input("sqrt(1-x)", label="ρ(x) = ")
    f_expr, f_lambda = function_input("sin(x)")
    a, b = segment_input(default_a=0.0, default_b=1.0)
    st.form_submit_button()

N = st.sidebar.number_input(min_value=1, label='N: кол-во узлов')

domain = Segment(a, b)

st.write(ro_expr * f_expr)

mu = []

for i in range(2 * N):
    mu.append(integrate(ro_expr * x_symbol ** i, domain))

print("Moments calculated")

st.write("## Моменты")

st.dataframe(mu)

st.write("## Линейное уравнение")


def mat_row(i):
    return [mu[j + i] for j in range(N - 1, -1, -1)]


A = np.array([mat_row(i) for i in range(0, N)])
b = np.array([-mu[i] for i in range(N, 2 * N)])
x = st_solve_linalg(A, b)

st.write("## Ортогональный многочлен ")

poly_coefs_from_smaller = np.concatenate(([1], x))[::-1]
poly = Polynomial(poly_coefs_from_smaller, domain=domain.to_tuple(), window=domain.to_tuple())
stpoly_expr = sp.Poly(reversed(poly_coefs_from_smaller), x_symbol).as_expr()
st.write(stpoly_expr)

poly_roots = poly.roots()

st.write("Корни")
st.write(poly_roots)

if np.allclose(poly(poly_roots), 0):
    st.success("Корни корректны")
else:
    st.error("Корни найдены не корректно")

if any(isinstance(root, complex) for root in poly_roots):
    st.error("Есть комплексные корни")

if len(poly_roots) != N:
    st.success("Количество корней != N")

st.write("## КФ")

# coefs
qf_nodes = poly_roots
qf_coefs = st_solve_linalg(
    np.array([poly_roots ** i for i in range(N)]),
    np.array(mu[:N])
)

# plot qf
fig, ax = plt.subplots()
plot_qf(qf_nodes, qf_coefs, ax)

x, y = poly.linspace(100)
ax.plot(x, y, label='Орт. многочлен')
ax.legend()
ax.set_xlim(domain.to_tuple())
st.pyplot(fig)

with st.expander(label='Более подробно'):
    gauss_df = pd.DataFrame({"Узлы": qf_nodes, "Коэф.": qf_coefs})
    st.dataframe(gauss_df)

if np.isclose(sum(qf_coefs), integrate(ro_expr, domain)):
    st.success("Сумма коэфициентов равна интегралу веса")
else:
    st.error("Сумма коэфициентов НЕ равна интегралу веса")

st.write("## Результат")


J = integrate(ro_expr * f_expr, domain)
J_approx = calc_qf(f_lambda, qf_nodes, qf_coefs)

st.latex(f"J = {J}")
st.latex(f"J_{{approx}} = {J_approx}")
st.latex(f"Abs.Error = {err(J, J_approx)}")
st.latex(f"Rel.Error = {rel_err(J, J_approx)}\%")
