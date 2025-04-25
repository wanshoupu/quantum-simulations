from typing import Union, Optional

import cirq
from cirq import EigenGate
from typing_extensions import override

from quompiler.circuits.circuit_builder import CircuitBuilder
from quompiler.construct.types import UnivGate
from quompiler.construct.cmat import UnitaryM, CUnitary


class CirqBuilder(CircuitBuilder):
    __UNIV_GATES = {
        # identity gate for completeness
        UnivGate.I: cirq.I,
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
    def get_univ_gate(self, m: UnitaryM) -> Optional[EigenGate]:
        univ_gate = UnivGate.get(m.matrix)
        if univ_gate:
            return CirqBuilder.__UNIV_GATES[univ_gate]

    @override
    def build_gate(self, m: UnitaryM):
        if isinstance(m, CUnitary):
            # TODO add gate approximation
            gate = self.get_univ_gate(m) or cirq.MatrixGate(m.matrix)
            target = [self.qubits[i] for i, c in enumerate(m.controls) if c is None]
            control = [self.qubits[i] for i, c in enumerate(m.controls) if c is not None]
            control_values = [int(c) for c in m.controls if c is not None]
            self.circuit.append(gate(*target).controlled_by(*control, control_values=control_values))
            return

        # TODO add KronUnitaryM gate
        custom_gate = cirq.MatrixGate(m.inflate())
        self.circuit.append(custom_gate(*self.qubits))

    @override
    def finish(self) -> cirq.Circuit:
        return self.circuit
