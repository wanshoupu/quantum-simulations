from typing import Union

import cirq
from cirq import EigenGate
from typing_extensions import override

from common.circuits.circuit_builder import CircuitBuilder
from common.construct.cmat import UnitaryM, CUnitary, UnivGate


class CirqBuilder(CircuitBuilder):
    __UNIV_GATES = {
        # necessary gates
        UnivGate.H: cirq.H,
        UnivGate.S: cirq.S,
        UnivGate.T: cirq.T,
        # additional auxiliary gates
        UnivGate.X: cirq.X,
        UnivGate.Y: cirq.Y,
        UnivGate.Z: cirq.Z,
    }

    def __init__(self, dimension: int):
        self.qubits = cirq.LineQubit.range(dimension)
        self.circuit = cirq.Circuit()

    @override
    def get_univ_gate(self, m: UnitaryM) -> Union[EigenGate, None]:
        univ_gate = UnivGate.get(m.matrix)
        if univ_gate:
            return CirqBuilder.__UNIV_GATES[univ_gate]

    @override
    def build_gate(self, m: UnitaryM):
        if isinstance(m, CUnitary):
            gate = self.get_univ_gate(m)
            if gate:
                control = [self.qubits[i] for i, c in enumerate(m.controls) if c is not None]
                control_values = [int(c) for c in m.controls if c is not None]
                self.circuit.append(gate(self.qubits[0]).controlled_by(*control, control_values=control_values))
                return

        custom_gate = cirq.MatrixGate(m.matrix)
        self.circuit.append(custom_gate(self.qubits[0]))

    @override
    def finish(self) -> cirq.Circuit:
        return self.circuit
