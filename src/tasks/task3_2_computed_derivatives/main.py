import streamlit as st
import pandas as pd
from sympy import parse_expr, diff, lambdify
import numpy as np
import sys


sys.path.append('')
sys.path.append('../../..')

from src.common.streamlit import function_input
from src.tasks.task3_2_computed_derivatives.derivatives import compute_derivatives

pd.set_option('precision', 10)
pd.set_option('expand_frame_repr', True)
pd.set_option('display.precision', 10)

DEFAULT_FUNCTION = 'exp(1.5 * x)'

st.write("""
# Численное дифференцирование. Вариант 10
""")

with st.form('base data'):
    f, f_lambda = function_input(DEFAULT_FUNCTION)
    a = st.number_input('a = ', value=1.2)
    m = st.number_input('number of nodes = ', value=3, min_value=3) - 1
    h = st.number_input('step = ', value=0.1, min_value=0.0)
    st.form_submit_button()

df = diff(f)
ddf = diff(df)

df_lambda = lambdify("x", df)
ddf_lambda = lambdify("x", ddf)

st.write(f"f'(x) = {df}")
st.write(f"f''(x) = {ddf}")

st.write("**Given data**")

df_given = pd.DataFrame(columns=["x", "f(x)", "f'(x)", "f''(x)"])
df_given['x'] = np.arange(start=a, stop=a + h * (m + 1), step=h)
df_given['f(x)'] = df_given.x.apply(lambda x: f_lambda(x))
df_given["f'(x)"] = df_given.x.apply(lambda x: df_lambda(x))
df_given["f''(x)"] = df_given.x.apply(lambda x: ddf_lambda(x))

st.dataframe(df_given)

st.write("**Approximated derivatives**")

df_calculated = df_given.copy()
df_calculated["approx f'"] = compute_derivatives(df_given, x='x', y='f(x)', step=h)
df_calculated["error f'"] = (df_calculated["f'(x)"] - df_calculated["approx f'"]).apply(lambda x: abs(x))
df_calculated["approx f''"] = compute_derivatives(df_calculated, x='x', y="approx f'", step=h)
df_calculated["error f''"] = (df_calculated["f''(x)"] - df_calculated["approx f''"]).apply(lambda x: abs(x))

st.dataframe(df_calculated[['x', 'f(x)', "approx f'", "error f'", "approx f''", "error f''"]].style.format("{:.10f}"))

for row in df_calculated[['x', 'f(x)', "approx f'", "error f'", "approx f''", "error f''"]].iterrows():
    st.write(row)
