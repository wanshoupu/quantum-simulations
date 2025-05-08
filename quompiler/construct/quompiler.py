from typing import Union

from numpy.typing import NDArray

from quompiler.construct.bytecode import Bytecode
from quompiler.construct.cgate import ControlledGate
from quompiler.utils.mat_utils import validm2l
from quompiler.construct.unitary import UnitaryM
from quompiler.utils.cnot_decompose import cnot_decompose
from quompiler.utils.mat2l_decompose import mat2l_decompose


def quompile(u: NDArray) -> Bytecode:
    s = u.shape
    um = UnitaryM(s[0], tuple(range(s[0])), u)
    return _quompile(um)


def _quompile(u: Union[UnitaryM, ControlledGate]) -> Bytecode:
    coms = _decompose(u)
    if len(coms) == 1:
        return Bytecode(coms[0])
    root = Bytecode(u)
    for c in coms:
        child = _quompile(c)
        root.append(child)
    return root


def _decompose(u: UnitaryM):
    if validm2l(u.matrix):
        return cnot_decompose(u)
    return mat2l_decompose(u)
