def compute_derivative_table_start(f_value, f_next_value, f_next_next_value, step):
    return (-3 * f_value + 4 * f_next_value - f_next_next_value) / (2 * step)


def compute_derivative_table_mid(f_prev_value, f_next_value, step):
    return (f_next_value - f_prev_value) / (2 * step)


def compute_derivative_table_end(f_value, f_prev_value, f_prevprev_value, step):
    return (3 * f_value - 4 * f_prev_value + f_prevprev_value) / (2 * step)


def compute_derivatives(df, step, x='x', y='f(x)'):
    assert len(df) > 2
    results = []
    # i = 0, first row
    results.append(compute_derivative_table_start(df.loc[0, y], df.loc[1, y], df.loc[2, y], step))
    # i = 1...(len-2)
    for i in range(1, len(df) - 1):
        results.append(compute_derivative_table_mid(df.loc[i-1, y], df.loc[i+1, y], step))
    # i = len - 1, last row
    last_i = len(df) - 1
    results.append(compute_derivative_table_end(df.loc[last_i, y], df.loc[last_i - 1, y], df.loc[last_i - 2, y], step))
    return results
