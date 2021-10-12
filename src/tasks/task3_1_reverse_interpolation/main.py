import streamlit as st
import sys
from sympy import parse_expr, lambdify
import pandas as pd
import plotly.express as px
import numpy as np


sys.path.append('')
sys.path.append('../../..')

from src.common.processes.find_roots import find_roots
from src.tasks.task1_find_roots.root_approximators.bisection_approximator import BisectionApproximator
from src.common.func import Func
from src.common.processes.function_nodes import df_from_points
from src.common.segment import Segment
from src.tasks.task3_1_reverse_interpolation.preparation import evenly_divide_into_nodes
from src.common.point import Point2D
from src.tasks.task2_interpolation.interpolators.lagrangian_interpolator import LagrangianInterpolator
from src.tasks.task2_interpolation.interpolators.newton_interpolator import NewtonInterpolator, \
    ReusableNewtonInterpolator
from src.tasks.task2_interpolation.preparation import randomly_select_nodes

DEFAULT_FUNCTION = 'cos(x) + 2*x'

st.write("""
# Обратная интерполяция. Вариант 10
""")

with st.form(key='func_form'):
    function_text = st.text_input(label='f(x) = ', value=DEFAULT_FUNCTION)
    a, b = st.slider('[a, b]', -5.0, 5.0, (0.5, 1.8), step=0.1)
    n = st.number_input('Number of nodes', min_value=10, max_value=100)
    submit_button = st.form_submit_button(label='Submit')

domain = Segment(a, b)

try:
    f = Func.from_string(function_text, domain)
except Exception as e:
    st.error(f"Invalid function entered: {e}")
    function_text = DEFAULT_FUNCTION
    f = Func.from_string(function_text, Segment(a, b))

xs = domain.evenly_divide_into_nodes(n)
points = f.calculate_points(xs)

df_points = df_from_points(points).rename(columns={'y':'f(x)'})

fig = px.line(df_points, x="x", y="f(x)", text="x")
fig.update_traces(textposition="bottom right")

st.plotly_chart(fig)

st.dataframe(df_points)

with st.form(key='F'):
    F = st.number_input('F =', value=2.4, format='%.4f')
    st.write("We will find x such that f(x) = F.")
    submit_button = st.form_submit_button(label='Submit')


st.write("""
## Method 1. Inverse function

Suppose the inverse is defined
""")

df1 = df_points.copy().rename(columns={'x': 'f^-1(F)', 'f(x)': 'F'})
df1 = df1[['F', 'f^-1(F)']].copy()

st.dataframe(df1)

with st.form('degree'):
    degree = st.number_input('Polynomial degree', min_value=1, max_value=(n - 1))
    st.form_submit_button(label='Submit')


df1_distance = df1.copy()
df1_distance['distance'] = df1_distance['F'].apply(lambda this_F: abs(this_F - F))
sorted_nodes = df1_distance.sort_values(by='distance').reset_index(drop=True)

selected_nodes_df = sorted_nodes.iloc[:degree + 1]

st.write("Selected nodes")
selected_nodes = selected_nodes_df[['F', 'f^-1(F)']].apply(lambda row: Point2D(row[0], row[1]), axis=1).to_list()

st.write(selected_nodes)

st.write("**Results with Newton Interpolator**")

interpolator = ReusableNewtonInterpolator(selected_nodes)
approx_x = interpolator.calc_approximate_value(F)
st.write(f"Approx. value: {approx_x}")
st.write(f"Absolute error: {abs(f.f(approx_x) - F)}")

# dfff1 = pd.DataFrame(columns=['x', 'f(x)'])
# dfff1['f(x)'] = np.arange(-10, 10, 0.01)
# dfff1['x'] = dfff1['f(x)'].apply(lambda y: interpolator.calc_approximate_value(y))
#
# fig = px.line(dfff1[(a <= dfff1.x) & (dfff1.x <= b)], x="x", y="f(x)")
# st.plotly_chart(fig)

st.write("""
## Method 2. Interpolate then find roots
""")

st.write("All nodes")
st.write(df_points)

with st.form('method2'):
    degree2 = st.number_input('Polynomial degree', min_value=1, max_value=(n - 1))
    epsilon = st.number_input('Epsilon', min_value=0.0000000000000001, max_value=1.0, value=0.001, step=0.000000000001, format='%.15f')
    st.form_submit_button(label='Submit')

selected_points2 = points[:degree2 + 1]
st.write("Selected points")
st.write(selected_points2)

interpolator2 = ReusableNewtonInterpolator(selected_points2)

st.write("Interpolation result:")

# Graph the interpolated polynomial
# graph_points = df_from_points([Point2D(x, interpolator2.calc_approximate_value(x)) for x in np.arange(-10, 10, 0.1)])
# fig = px.line(graph_points, x="x", y="y")
# fig.update_traces(textposition="bottom right")
# st.plotly_chart(fig)

interesting_function = Func(lambda x: interpolator2.calc_approximate_value(x) - F, None, domain)
roots = find_roots(interesting_function, epsilon=epsilon)

st.write("Roots")

for root in roots:
    st.write(f"Approx. value: {root}")
    st.write(f"Absolute error: {abs(f.f(root) - F)}")

