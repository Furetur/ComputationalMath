import sys
import streamlit as st
import sympy
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('')
sys.path.append('../../..')

from src.common.streamlit import function_input, segment_input


st.write("# 4.1 Приближенное вычисление интеграла по квадратурным формулам")

with st.form('main'):
    f_expr, f_lambda = function_input("x + sin(x) + 1")
    a, b = segment_input(default_a=0.0, default_b=1.0)
    st.form_submit_button()

st.write("## Правдивое значение интеграла")

integral_latex_string = r'\int_{' + str(a) + r'}^{' + str(b) + r'} f \,dx'

true_integral = sympy.integrate(f_expr, (sympy.Symbol('x'), a, b))

st.latex(integral_latex_string + f' = {true_integral}')


def rectangle_approximation(x, A):
    approx_value = A * f_lambda(x)
    st.latex(integral_latex_string + fr' \approx {A} * f({x}) = {A * f_lambda(x)}')
    error = abs(true_integral - approx_value)
    st.latex("Error:" + fr'''| {integral_latex_string} - {A} * f({x}) | = ''' + str(error))

    xlinspace = np.linspace(a, b, 100)
    fig, ax = plt.subplots()
    # function plot
    ax.plot(xlinspace, f_lambda(xlinspace), 'b', label='f(x)')
    # rectangle
    ax.fill([a, a, a + A, a + A], [0, f_lambda(x), f_lambda(x), 0], 'lightcoral', label='rectangle')
    # the point (x, f(x))
    ax.plot([x], [f_lambda(x)], 'go', label='node')
    ax.legend()
    st.pyplot(fig)


st.write("## Формула левого прямоугольника")

rectangle_approximation(a, b - a)

st.write("## Формула правого прямоугольника")

rectangle_approximation(b, b - a)

st.write("## Формула среднего прямоугольника")

rectangle_approximation((a + b) / 2, b - a)

st.write("## Формула трапеции")

trapezoid_approx = (b - a) * (f_lambda(a) + f_lambda(b)) / 2

st.latex(integral_latex_string + fr' \approx 0.5 * ({b} - {a}) * (f({a}) + f({b})) = {trapezoid_approx}')

trapezoid_error = abs(true_integral - trapezoid_approx)
st.latex("Error:" + fr'''| {integral_latex_string} - approx | = ''' + str(trapezoid_error))

xlinspace = np.linspace(a, b, 100)
fig, ax = plt.subplots()
# function plot
ax.plot(xlinspace, f_lambda(xlinspace), 'b', label='f(x)')
# trapezoid
if f_lambda(a) * f_lambda(b) >= 0:
    ax.fill([a, a, b, b], [0, f_lambda(a), f_lambda(b), 0], 'lightcoral', label='trapezoid')
# the point (x, f(x))
ax.plot([a, b], [f_lambda(a), f_lambda(b)], 'go', label='nodes')
ax.legend()
st.pyplot(fig)

st.write("## Формула Симпсона")

sim_approx = (b - a) / 6 * (f_lambda(a) + 4 * f_lambda((a + b) / 2) + f_lambda(b))

st.latex(integral_latex_string + r' \approx \frac{b - a}{6}(f(a) + f(\frac{a + b}{2}) + f(b)) = ' + str(sim_approx))

sim_error = abs(true_integral - sim_approx)

st.latex("Error:" + fr'''| {integral_latex_string} - approx | = ''' + str(sim_error))

st.write("## Формула 3/8")

three8_h = (b - a) / 3
three8_approx = (b - a) * (1 / 8 * f_lambda(a) + 3 / 8 * f_lambda(a + three8_h) + 3 / 8 * f_lambda(
    a + 2 * three8_h) + 1 / 8 * f_lambda(b))
three8_err = abs(true_integral - three8_approx)

st.latex(
    integral_latex_string + r' \approx (b-a)(\frac{1}{8}*f(a) + \frac{3}{8} * f(a+h) + \frac{3}{8} * f(a + 2h) + \frac{1}{8} * f(b)) = ' + str(
        three8_approx))
st.latex("Error:" + fr'''| {integral_latex_string} - approx | = ''' + str(three8_err))
