from typing import List

from src.common.qf.gauss import gauss_qf
from src.common.qf.qf_utils import remap_qf, calc_qf
from src.common.segment import Segment


def calc_complex_gauss(n: int, partitions: List[Segment], f_lambda) -> float:
    orig_nodes, orig_coefs = gauss_qf(n)
    result = 0
    for p in partitions:
        cur_nodes, cur_coefs = remap_qf(orig_nodes, orig_coefs, from_segment=Segment(-1, 1), to_segment=p)
        result += calc_qf(f_lambda, cur_nodes, cur_coefs)
    return result
