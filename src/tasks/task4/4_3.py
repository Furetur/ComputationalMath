import sys
from typing import List, Callable, Protocol

import pandas as pd
import streamlit as st
import seaborn as sns
import sympy
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('')
sys.path.append('../../..')


from src.common.segment import Segment
from src.common.streamlit import function_input, segment_input

sns.set_theme(style="darkgrid")

st.write("# 4.3 Сравнение погрешностей СКФ и метод Рунге")

with st.form('main'):
    f_expr, f_lambda = function_input("5*x + sin(5*x) + 1")
    a, b = segment_input(default_a=0.0, default_b=15.0)
    m = st.number_input(label='Number of partitions', min_value=1, value=8)
    l = st.number_input(label='l', min_value=2, value=2)
    st.form_submit_button()

st.write("Extended number of partitions")
st.latex(f"m*l={m * l}")

domain = Segment(a, b)
partitions = domain.split(m)
xpartitions = domain.split(m * l)
h = partitions[0].len()

st.write("## Правдивое значение интеграла")

integral_latex_string = r'\int_{' + str(a) + r'}^{' + str(b) + r'} f \,dx'
J = float(sympy.integrate(f_expr, (sympy.Symbol('x'), a, b)))
st.latex(integral_latex_string + f' = {J}')


# utils

def plot_function(ax):
    xlinspace = np.linspace(a, b, 100)
    ax.plot(xlinspace, [f_lambda(x) for x in xlinspace], 'b', label='f(x)')


def err(approx):
    return abs(J - approx)


def rel_err(approx):
    return err(approx) / abs(J) * 100


def runge(J_h, J_hl, d):
    r = d + 1
    return (l ** r * J_hl - J_h) / (l ** r - 1)


def st_report(J_h, J_hl, d):
    st.latex(f"J(h) = {J_h}")
    st.latex(f"Abs Error |J - J(h)| = {err(J_h)}")
    runged = runge(J_h, J_hl, d)
    st.latex(f"RungeJ(h, l) = {runged}")
    st.latex(f"Abs Error |J - RungeJ(h, l)| = {err(runged)}")


ComplexApprox = Callable[[List[Segment]], float]


class Approx(Protocol):
    name: str
    d: int

    def calc(self, parts: List[Segment]) -> float: ...


def report(approx: Approx):
    J_h = approx.calc(partitions)
    J_hl = approx.calc(xpartitions)
    st_report(J_h, J_hl, approx.d)


class RectApprox(Approx):
    def __init__(self, kind: str):
        if kind == 'left':
            self.d = 0
            self.name = 'Left Rect'
            self.get_node = lambda seg: f_lambda(seg.start)
        elif kind == 'right':
            self.d = 0
            self.name = 'Right Rect'
            self.get_node = lambda seg: f_lambda(seg.end)
        elif kind == 'mid':
            self.d = 1
            self.name = 'Mid Rect'
            self.get_node = lambda seg: f_lambda(seg.mid())
        else:
            raise Exception("This kind is not supported")

    def calc(self, parts: List[Segment]) -> float:
        return parts[0].len() * sum(self.get_node(p) for p in parts)


class TrapezeApprox(Approx):
    d = 1
    name = 'Trapezoid'

    def calc(self, parts: List[Segment]) -> float:
        return parts[0].len() / 2 * sum(f_lambda(p.start) + f_lambda(p.end) for p in parts)


class SimpApprox(Approx):
    d = 3
    name = 'Simpson'

    def calc(self, parts: List[Segment]) -> float:
        return parts[0].len() / 6 * sum(
            f_lambda(p.start) + 4 * f_lambda((p.start + p.end) / 2) + f_lambda(p.end) for p in parts)


# plot

fig, ax = plt.subplots()
plot_function(ax)
st.pyplot(fig)


def df_row_approx(approx: Approx):
    J_h = approx.calc(partitions)
    J_hl = approx.calc(xpartitions)
    rng = runge(J_h, J_hl, approx.d)
    return approx.name, J_h, err(J_h), J_hl, err(J_hl), rng, err(rng)


approxs = [
    RectApprox(kind='left'),
    RectApprox(kind='right'),
    RectApprox(kind='mid'),
    TrapezeApprox(),
    SimpApprox()
]

st.write('## Результаты')

data = dict()

for approx in approxs:
    J_h = approx.calc(partitions)
    J_hl = approx.calc(xpartitions)
    rng = runge(J_h, J_hl, approx.d)
    data[approx.name] = (J_h, err(J_h), J_hl, err(J_hl), rng, err(rng))

df = pd.DataFrame.from_dict(
    data,
    orient='index',
    columns=['J(h)', 'Err. J(h)', 'J(h/l)', 'Err. J(h/l)', 'Runge', 'Err. Runge']
)

st.table(df.style.format("{:.10f}"))

st.write('## Сравнение относительных погрешностей')

data_errors = dict()

for approx in approxs:
    J_h = approx.calc(partitions)
    J_hl = approx.calc(xpartitions)
    rng = runge(J_h, J_hl, approx.d)
    data_errors[approx.name] = (rel_err(J_h), rel_err(J_hl), rel_err(rng))

df_errors = pd.DataFrame.from_dict(
    data_errors,
    orient='index',
    columns=['Rel. Err. J(h)', 'Rel. Err. J(h/l)', 'Rel. Err. Runge']
)

st.table(df_errors.style.format("{:.10f}"))

f, ax = plt.subplots()

# visualization
df_err_comp = pd.DataFrame(columns=['Name', 'Rel. Err', 'Type'])

for i, (name, (J_h_relerr, J_hl_relerr, runge_relerr)) in enumerate(data_errors.items()):
    df_err_comp.loc[2 * i] = (name, J_hl_relerr, 'J(h/l)')
    df_err_comp.loc[2 * i + 1] = (name, runge_relerr, 'Runge')


sns.set_color_codes("deep")

g = sns.barplot(data=df_err_comp, x='Rel. Err', y='Name', hue='Type', ax=ax, color='g', palette='bright')
g.set_xscale("log")


ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 100), ylabel="",
       xlabel="Relative Error (lower is better)")
sns.despine(left=True, bottom=True)

st.pyplot(f)
