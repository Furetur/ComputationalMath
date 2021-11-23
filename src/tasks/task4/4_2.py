import sys
from typing import List

import streamlit as st
import sympy
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from sympy import diff, lambdify

sys.path.append('')
sys.path.append('../../..')

from src.common.utils import maximize

from src.common.segment import Segment
from src.common.streamlit import function_input, segment_input

st.write("# 4.2 Приближенное вычисление интеграла по составным квадратурным формулам")

with st.form('main'):
    f_expr, f_lambda = function_input("5*x + sin(5*x) + 1")
    a, b = segment_input(default_a=0.0, default_b=15.0)
    m = st.number_input(label='Number of partitions', min_value=1, value=8)
    st.form_submit_button()

df = lambdify("x", diff(f_expr))
ddf = lambdify("x", diff(diff(f_expr)))
ddddf = lambdify("x", diff(diff(diff(diff(f_expr)))))

_, df_abs_max = maximize(lambda x: abs(df(x)), a, b)
_, ddf_abs_max = maximize(lambda x: abs(ddf(x)), a, b)
_, ddddf_abs_max = maximize(lambda x: abs(ddddf(x)), a, b)

domain = Segment(a, b)
partitions = domain.split(m)
h = partitions[0].len()

with st.expander("Debug info"):
    st.latex(f"f' = {diff(f_expr)}")
    st.latex(f"f'' = {diff(diff(f_expr))}")
    st.latex(f"f'''' = {diff(diff(diff(diff(f_expr))))}")

    st.latex(f"max|f'| = {df_abs_max}")
    st.latex(f"max|f''| = {ddf_abs_max}")
    st.latex(f"max|f''''| = {ddddf_abs_max}")

st.write("## Правдивое значение интеграла")

integral_latex_string = r'\int_{' + str(a) + r'}^{' + str(b) + r'} f \,dx'
true_integral = float(sympy.integrate(f_expr, (sympy.Symbol('x'), a, b)))
st.latex(integral_latex_string + f' = {true_integral}')


def plot_function(ax):
    xlinspace = np.linspace(a, b, 100)
    ax.plot(xlinspace, [f_lambda(x) for x in xlinspace], 'b', label='f(x)')


def draw_rect_for_approx(x: float, A: float, segment: Segment, ax):
    a, b = segment.start, segment.end
    # rectangle
    ax.fill([a, a, a + A, a + A], [0, f_lambda(x), f_lambda(x), 0], 'lightcoral')
    # the point (x, f(x))
    ax.plot([x], [f_lambda(x)], 'go')


def draw_complex_rectangle(xs: List[float]):
    assert len(xs) == len(partitions)
    # plot
    fig, ax = plt.subplots()
    # draw function
    plot_function(ax)
    # draw rectangles
    for x, partition in zip(xs, partitions):
        draw_rect_for_approx(x, h, partition, ax)
    ax.legend()
    st.pyplot(fig)


st.write("## Формула левого прямоугольника")

xs = [p.start for p in partitions]
approx_value = sum(h * f_lambda(x) for x in xs)

st.latex(fr"{integral_latex_string} \approx {h} * \sum_{{j=0}}^{{m-1}} f(x_{{j}}) = {approx_value}")
st.latex(fr"Theoretical Error <= \frac{{(B-A)^2}}{{2m}} * max|f'| = {(b - a) ** 2 / (2 * m) * df_abs_max}")
st.latex(fr"Absolute Error = | {integral_latex_string} - approx | = {abs(true_integral - approx_value)}")

draw_complex_rectangle(xs)

st.write("## Формула правого прямоугольника")

xs = [p.end for p in partitions]
approx_value = sum(h * f_lambda(x) for x in xs)

st.latex(fr"{integral_latex_string} \approx {h} * \sum_{{j=0}}^{{m-1}} f(x_{{j}}) = {approx_value}")
st.latex(fr"Theoretical Error <= \frac{{(B-A)^2}}{{2m}} * max|f'| = {(b - a) ** 2 / (2 * m) * df_abs_max}")
st.latex(fr"Absolute Error = | {integral_latex_string} - approx | = {abs(true_integral - approx_value)}")

draw_complex_rectangle(xs)

st.write("## Формула среднего прямоугольника")

xs = [(p.start + p.end) / 2 for p in partitions]
approx_value = sum(h * f_lambda(x) for x in xs)

st.latex(fr"{integral_latex_string} \approx {h} * \sum_{{j=0}}^{{m-1}} f(x_{{j}}) = {approx_value}")
st.latex(fr"Theoretical Error <= \frac{{(B-A)^3}}{{24m^2}} * max|f''| = {(b - a) ** 3 / (24 * m ** 2) * ddf_abs_max}")
st.latex(fr"Absolute Error = | {integral_latex_string} - approx | = {abs(true_integral - approx_value)}")

draw_complex_rectangle(xs)

st.write("## Формула трапеции")

approx_value = h / 2 * sum(f_lambda(p.start) + f_lambda(p.end) for p in partitions)

st.latex(fr"{integral_latex_string} \approx \frac{{h}}{{2}} * (f(a) + 2f(a+h) + ... + f(b)) = {approx_value}")
st.latex(fr"Theoretical Error <= \frac{{(B-A)^3}}{{12m^2}} * max|f''| = {(b - a) ** 3 / (12 * m ** 2) * ddf_abs_max}")
st.latex(fr"Absolute Error = | {integral_latex_string} - approx | = {abs(true_integral - approx_value)}")

# plot
fig, ax = plt.subplots()
# draw function
plot_function(ax)
# draw nodes
nodes = [p.start for p in partitions] + [b]
ax.plot(nodes, [f_lambda(x) for x in nodes], 'go', label='nodes')
ax.fill([a] + nodes + [b], [0] + [f_lambda(x) for x in nodes] + [0], 'lightcoral', label='trapezoid')

ax.legend()
st.pyplot(fig)

st.write("## Формула Симпсона")

approx_value = h / 6 * sum(f_lambda(p.start) + 4 * f_lambda((p.start + p.end) / 2) + f_lambda(p.end) for p in partitions)

st.latex(r"Simple formula = \int_{a}^{b} f dx \approx \frac{b - a}{6}(f(a) + 4f(\frac{a + b}{2}) + f(b))")
st.latex(
    integral_latex_string + r"\approx \frac{h}{6} \sum (f(y_j) + 4f(\frac{y_j + y_{j+1}}{2}) + f(y_{j+1})) = " + str(
        approx_value))
st.latex(
    fr"Theoretical Error <= \frac{{(B-A)^5}}{{2880m^4}} * max|f''''| = {(b - a) ** 5 / (2880 * m ** 4) * ddddf_abs_max}")
st.latex(fr"Absolute Error = | {integral_latex_string} - approx | = {abs(true_integral - approx_value)}")

# plot
fig, ax = plt.subplots()
# draw function
plot_function(ax)
# draw nodes
triplets = [(p.start, (p.start + p.end) / 2, p.end) for p in partitions]
nodes = []
for p in partitions:
    nodes.append(p.start)
    nodes.append(p.mid())
nodes.append(b)
ax.plot(nodes, [f_lambda(x) for x in nodes], 'go', label='nodes')
# draw parabolas
parabola_xs = []
parabola_ys = []
for triplet in triplets:
    parabola_f = interpolate.interp1d(triplet, [f_lambda(x) for x in triplet], kind='quadratic')
    for x in np.linspace(triplet[0], triplet[2], 10):
        parabola_xs.append(x)
        parabola_ys.append(parabola_f(x))
ax.fill([a] + parabola_xs + [b], [0] + parabola_ys + [0], 'lightcoral', label='parabola')

ax.legend()
st.pyplot(fig)
