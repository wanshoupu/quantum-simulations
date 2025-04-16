import cirq
import numpy as np
from typing_extensions import override

from common.construct.circuit import CircuitBuilder
from common.construct.cmat import UnitaryM, validm2l, immutable


class CirqBuilder(CircuitBuilder):
    __UNIV_GATES = {
        ((0j, 1), (1, 0j)): cirq.X,
        ((0j, 1j), (-1j, 0j)): cirq.Y,
        ((1, 0j), (0j, -1)): cirq.Z,
        ((1, 0j), (0j, 1j)): cirq.S,
        ((1, 0j), (0j, np.exp(1j * np.pi / 4))): cirq.T,
        immutable(np.array([[1, 1], [1, -1]]) / np.sqrt(2)): cirq.H,
    }

    def __init__(self, dimension: int):
        super().__init__(dimension)
        self.qubits = cirq.LineQubit.range(dimension)
        self.circuit = cirq.Circuit()

    @override
    def get_unigate(self, m: UnitaryM) -> object:
        pass

    @override
    def build_gate(self, m: UnitaryM):
        if not validm2l(m.matrix):
            custom_gate = cirq.MatrixGate(m.matrix)
            self.circuit.append(custom_gate(self.qubits[0]))
        # if isinstance(m, CUnitary):

    @override
    def finish(self) -> cirq.Circuit:
        pass
