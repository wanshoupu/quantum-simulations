from numpy.typing import NDArray

from quompiler.construct.bytecode import Bytecode
from quompiler.construct.cmat import validm2l, UnitaryM, CUnitary
from quompiler.utils.cnot_decompose import cnot_decompose
from quompiler.utils.mat2l_decompose import mat2l_decompose


def quompile(u: NDArray) -> Bytecode:
    s = u.shape
    um = UnitaryM(s[0], u, indexes=tuple(range(s[0])))
    return _quompile(um)


def _quompile(u: UnitaryM) -> Bytecode:
    if u.issinglet():
        return Bytecode(CUnitary.convert(u))
    root = Bytecode(u)
    coms = _decompose(u)
    for c in coms:
        child = _quompile(c)
        root.append(child)
    return root


def _decompose(u: UnitaryM):
    if validm2l(u.matrix):
        return cnot_decompose(u)
    return mat2l_decompose(u)
