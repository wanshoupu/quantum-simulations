"""
This module provide the compilation functionalities.
If needed, it may make distinctions between target qubits and ancilla qubits.
"""
from typing import Union

from numpy.typing import NDArray

from quompiler.circuits.circuit_builder import CircuitBuilder
from quompiler.construct.bytecode import Bytecode, ReverseBytecodeIter
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.unitary import UnitaryM
from quompiler.utils.cnot_decompose import cnot_decompose
from quompiler.utils.mat2l_decompose import mat2l_decompose
from quompiler.utils.mat_utils import validm2l


class Quompiler:
    def __init__(self, flags):
        self.flags = flags


def quompile(u: NDArray) -> Bytecode:
    s = u.shape
    um = UnitaryM(s[0], tuple(range(s[0])), u)
    return _quompile(um)


def _quompile(u: Union[UnitaryM, CtrlGate]) -> Bytecode:
    coms = _decompose(u)
    if len(coms) == 1:
        return Bytecode(coms[0])
    root = Bytecode(u)
    for c in coms:
        child = _quompile(c)
        root.append(child)
    return root


def _decompose(u: UnitaryM):
    if isinstance(u, UnitaryM):
        mat = u.matrix
    elif isinstance(u, CtrlGate):
        return [u]
    else:
        raise TypeError("Unsupported unitary matrix")
    if validm2l(mat):
        return cnot_decompose(u)
    return mat2l_decompose(u)


class CircuitInterp:

    def __init__(self, builder: CircuitBuilder):
        self.builder = builder

    def interpret(self, u: NDArray):
        component = quompile(u)
        for c in ReverseBytecodeIter(component):
            m = c.data
            if isinstance(m, CtrlGate):
                # TODO for now draw single-qubit + controlled single-qubit as gate.
                # TO BE breakdown further to elementary gates only
                self.builder.build_gate(m)
            elif isinstance(m, UnitaryM):
                self.builder.build_group(m)
