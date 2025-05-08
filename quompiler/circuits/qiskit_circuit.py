from typing import Optional

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from typing_extensions import override

from quompiler.circuits.circuit_builder import CircuitBuilder
from quompiler.utils.mat_utils import validm2l
from quompiler.construct.unitary import UnitaryM
from quompiler.construct.types import UnivGate


class QiskitBuilder(CircuitBuilder):
    def __init__(self, dimension: int):
        self.qubits = []
        self.circuit = qiskit.QuantumCircuit()

    @override
    def get_univ_gate(self, m: UnitaryM) -> Optional[UnivGate]:
        pass

    @override
    def build_gate(self, m: UnitaryM):
        if not validm2l(m.matrix):
            custom_gate = UnitaryGate(m.matrix)
            self.circuit.append(custom_gate, self.qubits)

    @override
    def finish(self) -> QuantumCircuit:
        pass
