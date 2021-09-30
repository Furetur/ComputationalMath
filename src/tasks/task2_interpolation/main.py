import streamlit as st
import sys
from sympy import parse_expr, lambdify
import pandas as pd
import plotly.express as px

sys.path.append('')
sys.path.append('../../..')

from src.common.point import Point2D
from src.tasks.task2_interpolation.interpolators.lagrangian_interpolator import LagrangianInterpolator
from src.tasks.task2_interpolation.interpolators.newton_interpolator import NewtonInterpolator
from src.tasks.task2_interpolation.preparation import randomly_select_nodes

DEFAULT_FUNCTION = 'cos(x) + 2*x'

st.write("""
# Интерполяция. Вариант 10
""")

with st.form(key='func_form'):
    function_text = st.text_input(label='f(x) = ', value=DEFAULT_FUNCTION)
    a, b = st.slider('[a, b]', -5.0, 5.0, (0.5, 1.8), step=0.1)
    n = st.number_input('Number of nodes', min_value=10, max_value=100)
    submit_button = st.form_submit_button(label='Submit')

try:
    lambda_f = lambdify("x", parse_expr(function_text))
except Exception as e:
    st.error(f"Invalid function entered: {e}")
    function_text = DEFAULT_FUNCTION
    lambda_f = lambdify("x", parse_expr(function_text))

df = pd.DataFrame(columns=('x', 'f(x)'))
df['x'] = randomly_select_nodes(a, b, n)
df['f(x)'] = df.x.apply(lambda x: lambda_f(x))

fig = px.line(df, x="x", y="f(x)", text="x")
fig.update_traces(textposition="bottom right")

st.plotly_chart(fig)


with st.form('x'):
    x = st.number_input('x', value=1.2, format='%.4f')
    st.form_submit_button(label='Submit')

dff = df.copy()
dff['distance'] = dff['x'].apply(lambda this_x: abs(this_x - x))

st.write("All nodes")

sorted_nodes = dff.sort_values(by='distance').reset_index(drop=True)

st.dataframe(sorted_nodes)

with st.form('degree'):
    degree = st.number_input('Polynomial degree', min_value=1, max_value=(n - 1))
    st.form_submit_button(label='Submit')

selected_nodes_df = sorted_nodes.iloc[:degree + 1]

st.write("Selected nodes")
selected_nodes = selected_nodes_df[['x', 'f(x)']].apply(lambda row: Point2D(row[0], row[1]), axis=1).to_list()

st.write(selected_nodes)

# Interpolation
interpolators = [LagrangianInterpolator(), NewtonInterpolator()]

st.write("## Results")
st.write(f"Real value: f(x) = {lambda_f(x)}")

for interpolator in interpolators:
    st.write(f"## {interpolator.method_name}")
    approx_y = interpolator.calc_approximate_value(x, selected_nodes)
    st.write(f"Approx. value: {approx_y}")
    st.write(f"Absolute error: {abs(lambda_f(x) - approx_y)}")
