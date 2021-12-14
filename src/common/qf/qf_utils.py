from src.common.segment import Segment


def calc_qf(f_lambda, nodes, coefs):
    assert len(nodes) == len(coefs)
    return sum(A_k * f_lambda(x_k) for A_k, x_k in zip(coefs, nodes))


def remap_qf(nodes, coefs, from_segment: Segment, to_segment: Segment):
    q = to_segment.len() / from_segment.len()
    a = from_segment.start
    c = to_segment.start
    new_nodes = [c + q * (x_k - a) for x_k in nodes]
    new_coefs = [A_k * q for A_k in coefs]
    return new_nodes, new_coefs


def plot_qf(nodes, coefs, ax):
    assert len(coefs) == len(nodes)
    n = len(nodes)
    ax.plot(nodes, [0] * n, 'bo', label='Узлы')
    ax.plot(nodes, coefs, 'go', label='Коэфициенты')