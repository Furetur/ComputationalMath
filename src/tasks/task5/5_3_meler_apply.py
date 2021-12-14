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

from src.common.polynomial.chebyshev import chebyshevs
from src.common.qf.meller import meller_qf
from src.common.qf.qf_utils import calc_qf, remap_qf, plot_qf
from src.common.polynomial.legendre import legendres
from src.common.qf.gauss import gauss_qf
from src.common.segment import Segment
from src.common.streamlit import function_input, segment_input

st.write("# 5.3 Применение КФ Мелера и сравнение с Гаусса")

st.write("Хотим вычислить интеграл функции:")

st.latex(r"\int_{-1}^{1} \frac{1}{\sqrt{1-x^2}} * cos(x) (1 + x^2)")
J = 3.425419173884657470780366457489081410132175818243203051298256339765777946418377093416592960869349206

ro_lambda = lambda x: 1/np.sqrt(1 - np.power(x, 2))
f_lambda = lambda x: np.cos(x) * (1 + np.power(x, 2))
i_lambda = lambda x: ro_lambda(x) * f_lambda(x)

st.write("Правдивое значение (Wolfram Alpha):")
st.latex(J)

st.sidebar.latex(f"J = {J}")

Ns = [3, 6, 7, 8]
st.latex(f"N = {Ns}")

a, b = -1, 1

T = chebyshevs(max(Ns))

st.write("## Показать КФ Мелера")

N = st.number_input(label='N', value=5, min_value=1, max_value=max(Ns))
nodes, coefs = meller_qf(N)
fig, ax = plt.subplots()
plot_qf(nodes, coefs, ax)

ax.set_xlim([-1, 1])
x, y = T[N].linspace(100)
ax.plot(x, y, label='Чебышёв')
ax.legend()
st.pyplot(fig)

with st.expander(label="Более подробно"):
    meller_df = pd.DataFrame({"Nodes": nodes, "Coefs": coefs})
    st.dataframe(meller_df)


st.write("## Вычисление интеграла")

qfs = [meller_qf(N) for N in Ns]
J_approxs = [calc_qf(f_lambda, nodes, coefs) for (nodes, coefs) in qfs]
errs = [abs(J - J_approx) for J_approx in J_approxs]
rel_errs = [err / J * 100 for err in errs]

df = pd.DataFrame({
    "N": Ns,
    "Приближенное значение": J_approxs,
    "Абсолютная погрешность": errs,
    "Относительная погрешность": rel_errs
})

st.dataframe(df)

fig, ax = plt.subplots()
g = sns.barplot(data=df, x="Относительная погрешность", y="N", ax=ax, orient='h')
g.set_xscale("log")
ax.set_xlim([0, 100])
st.pyplot(fig)
