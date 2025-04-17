from numpy.typing import NDArray

from common.construct.bytecode import Bytecode
from common.construct.cmat import validm2l, UnitaryM
from common.utils.cnot_decompose import cnot_decompose
from common.utils.mat2l_decompose import mat2l_decompose


def quompile(u: NDArray) -> Bytecode:
    s = u.shape
    um = UnitaryM(s[0], u, indexes=tuple(range(s[0])))
    return _quompile(um)


def _quompile(u: UnitaryM) -> Bytecode:
    root = Bytecode(u)
    coms = _decompose(u)
    if len(coms) > 1:
        for c in coms:
            child = _quompile(c)
            root.append(child)
    else:
        # the tree rooted at root has a single leaf. So make the root itself a leaf
        root.data = coms[0]
    return root


def _decompose(u: UnitaryM):
    if validm2l(u.matrix):
        return cnot_decompose(u)
    return mat2l_decompose(u)
