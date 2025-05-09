"""
This module provide the compilation functionalities.
If needed, it may make distinctions between target qubits and ancilla qubits.
"""
from typing import Union

from numpy.typing import NDArray

from quompiler.construct.bytecode import Bytecode, ReverseBytecodeIter
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.std_gate import CtrlStdGate
from quompiler.construct.unitary import UnitaryM
from quompiler.qompile.configure import QompilerConfig, QompilePlatform
from quompiler.utils.cnot_decompose import cnot_decompose
from quompiler.utils.mat2l_decompose import mat2l_decompose
from quompiler.utils.mat_utils import validm2l


class CircuitInterp:

    def __init__(self, config: QompilerConfig):
        dim = config.device.dimension
        self.builder = QompilePlatform[config.target](dim)

    def interpret(self, u: NDArray):
        component = self.quompile(u)
        for c in ReverseBytecodeIter(component):
            m = c.data
            if isinstance(m, CtrlStdGate) or isinstance(m, CtrlGate):
                # TODO for now draw single-qubit + controlled single-qubit as gate.
                # TO BE breakdown further to elementary gates only
                self.builder.build_gate(m)
            elif isinstance(m, UnitaryM):
                self.builder.build_group(m)

    def quompile(self, u: NDArray) -> Bytecode:
        s = u.shape
        um = UnitaryM(s[0], tuple(range(s[0])), u)
        return self._quompile(um)

    def _quompile(self, u: Union[UnitaryM, CtrlGate]) -> Bytecode:
        coms = self._decompose(u)
        if len(coms) == 1:
            return Bytecode(coms[0])
        root = Bytecode(u)
        for c in coms:
            child = self._quompile(c)
            root.append(child)
        return root

    def _decompose(self, u: UnitaryM):
        if isinstance(u, UnitaryM):
            mat = u.matrix
        elif isinstance(u, CtrlGate):
            return [u]
        else:
            raise TypeError("Unsupported unitary matrix")
        if validm2l(mat):
            return cnot_decompose(u)
        return mat2l_decompose(u)

    def finish(self, optimized=False) -> object:
        return self.builder.finish(optimized=optimized)

    def all_qubits(self):
        return self.builder.all_qubits()
