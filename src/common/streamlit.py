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
