import streamlit as st
import numpy as np


def st_solve_linalg(A, b):
    st.write("Решим линейное уравнение")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Матрица коэф.")
        st.write(A)
    with col2:
        st.write("Правая часть")
        st.write(b)
    st.write(f"Вектор решений")
    x = np.linalg.solve(A, b)
    st.write(x)
    if np.allclose(np.dot(A, x), b):
        st.success("Решено корректно")
    else:
        st.error("Решено некорректно")
    return x
