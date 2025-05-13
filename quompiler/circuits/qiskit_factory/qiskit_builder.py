from typing import Union, Optional

from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from typing_extensions import override

import qiskit

from quompiler.circuits.qbuilder import CircuitBuilder
from quompiler.config.construct import DeviceConfig
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.std_gate import CtrlStdGate
from quompiler.construct.types import UnivGate
from quompiler.construct.unitary import UnitaryM
from quompiler.utils.mat_utils import validm2l


class QiskitBuilder(CircuitBuilder):
    def __init__(self, deviceConfig: DeviceConfig):
        self.qubits = []
        self.circuit = qiskit.QuantumCircuit()

    @override
    def get_univ_gate(self, m: Union[UnitaryM, CtrlGate, CtrlStdGate]) -> Optional[UnivGate]:
        pass

    @override
    def build_gate(self, m: Union[UnitaryM, CtrlGate, CtrlStdGate]):
        if not validm2l(m.matrix):
            custom_gate = UnitaryGate(m.matrix)
            self.circuit.append(custom_gate, self.qubits)

    @override
    def finish(self) -> QuantumCircuit:
        pass

    @override
    def all_qubits(self) -> list:
        pass
