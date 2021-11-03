import streamlit as st
from sympy import parse_expr, lambdify


def function_input(default_value: str):
    try:
        f_default = parse_expr(default_value)
        f_default_lambda = lambdify("x", f_default)
    except Exception as e:
        st.error(e)
        raise Exception("Given default function is invalid")

    function_text = st.text_input(label='f(x) = ', value=default_value)
    try:
        f = parse_expr(function_text)
        f_lambda = lambdify("x", f)
    except Exception as e:
        st.error(e)
        f = f_default
        f_lambda = f_default_lambda

    return f, f_lambda


def segment_input(default_a, default_b, min_a=None, min_b=None, step=0.01):
    if default_a >= default_b:
        raise Exception("default_a must be < default_b")

    col1, col2 = st.columns(2)

    with col1:
        a = st.number_input('a', value=default_a, min_value=min_a, step=step)

    with col2:
        b = st.number_input('b', value=default_b, min_value=min_b, step=step)

    if a >= b:
        st.error("a must be less than b. Now using default [a, b]")
        return default_a, default_b
    return a, b
