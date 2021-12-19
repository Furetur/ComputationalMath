from numpy.polynomial import Polynomial


def omega(roots):
    """
    Finds w_n = (x-x_1) * (x-x_2) * ... * (x-x_n),
    where roots = [x_1 ... x_n]
    """
    result = 1
    for x_i in roots:
        cur_poly = Polynomial([-x_i, 1])
        result *= cur_poly
    return result
