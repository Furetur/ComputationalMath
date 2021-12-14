from numpy.polynomial import Polynomial


def legendres(n: int):
    if n < 0:
        raise Exception("n must be >= 0")

    X = Polynomial([0, 1])
    P = [Polynomial([1]), X]
    # P_n = (2n - 1)/n * P_{n-1} * X - (n-1)/n * P_{n-2}
    for i in range(2, n + 1):
        P_i = (2 * i - 1) / i * P[i - 1] * X - (i - 1) / i * P[i - 2]
        P.append(P_i)
    return P[:n + 1]
