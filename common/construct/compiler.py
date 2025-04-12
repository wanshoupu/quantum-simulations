import numpy as np

from bytecode import Bytecode
from common.utils.cnot_decompose import cnot_decompose
from common.utils.mat2l import mat2l_decompose, validm2l
from common.utils.mgen import cyclic_matrix


def compile(u: np.ndarray) -> Bytecode:
    root = Bytecode(u)
    coms = _dec(u)
    if len(coms) > 1:
        for c in coms:
            child = compile(c)
            root.append(child)
    return root


def _dec(u):
    if validm2l(u):
        return cnot_decompose(u)
    return mat2l_decompose(u)


if __name__ == '__main__':
    u = cyclic_matrix(8, 1)
    bc = compile(u)
    print(bc)
