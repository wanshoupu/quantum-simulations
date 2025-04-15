from common.construct.bytecode import Bytecode
from common.utils.cnot_decompose import cnot_decompose
from common.utils.mat2l_decompose import mat2l_decompose
from common.construct.cmat import validm2l, UnitaryM
from numpy.typing import NDArray


def quompile(u: NDArray) -> Bytecode:
    s = u.shape
    um = UnitaryM(s[0], u, row_indexes=tuple(range(s[0])))
    return _quompile(um)


def _quompile(u: UnitaryM) -> Bytecode:
    root = Bytecode(u)
    coms = _dec(u)
    if len(coms) > 1:
        for c in coms:
            child = _quompile(c)
            root.append(child)
    return root


def _dec(u: UnitaryM):
    if validm2l(u.matrix):
        ms = cnot_decompose(u)
        c, v = ms[:-1], ms[-1]
        return c + (v,) + c[::-1]
    return mat2l_decompose(u)
