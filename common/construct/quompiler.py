from common.construct.bytecode import Bytecode
from common.utils.cnot_decompose import cnot_decompose
from common.utils.mat2l_decompose import mat2l_decompose
from common.construct.cmat import validm2l
from numpy.typing import NDArray


def quompile(u: NDArray) -> Bytecode:
    root = Bytecode(u)
    coms = _dec(u)
    if len(coms) > 1:
        for c in coms:
            child = quompile(c)
            root.append(child)
    return root


def _dec(u):
    if validm2l(u):
        ms = cnot_decompose(u)
        c, v = ms[:-1], ms[-1]
        return c + [v] + c[::-1]
    return mat2l_decompose(u)
