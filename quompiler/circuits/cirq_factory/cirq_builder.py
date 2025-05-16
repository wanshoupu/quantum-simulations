import warnings
from typing import Union, Optional

import cirq
from cirq import EigenGate, Circuit, merge_single_qubit_gates_to_phased_x_and_z, eject_z, drop_negligible_operations, drop_empty_moments

from quompiler.circuits.qbuilder import CircuitBuilder
from quompiler.circuits.qdevice import QDevice
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.std_gate import CtrlStdGate
from quompiler.construct.types import UnivGate, QType
from quompiler.construct.unitary import UnitaryM


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

    def __init__(self, device: QDevice):
        self.device = device
        self.circuit = cirq.Circuit()

    def get_univ_gate(self, m: Union[UnitaryM, CtrlGate, CtrlStdGate]) -> Optional[EigenGate]:
        if isinstance(m, CtrlGate) or isinstance(m, CtrlStdGate):
            matrix = m.gate.matrix if isinstance(m, CtrlStdGate) else m._unitary.matrix
            univ_gate = UnivGate.get(matrix)
            if univ_gate:
                return CirqBuilder.__UNIV_GATES[univ_gate]
        return None

    def build_gate(self, m: Union[UnitaryM, CtrlGate, CtrlStdGate]):
        if isinstance(m, CtrlGate):
            gate = self.get_univ_gate(m) or cirq.MatrixGate(m._unitary.matrix)
            self._append_gate(m.controls, gate, m.qids())
        if isinstance(m, CtrlStdGate):
            self._append_gate(m.get_controller(), m.gate, m.get_qubits())
        warnings.warn(f"Warning: gate of type {type(m)} is ignored.")

    def _append_gate(self, controller, gate, qids):
        target = [self.device.map(qids[i]) for i, c in enumerate(controller) if c is QType.TARGET]
        control = [self.device.map(qids[i]) for i, c in enumerate(controller) if c in QType.CONTROL0 | QType.CONTROL1]
        control_values = [c.base[0] for c in controller if c in QType.CONTROL0 | QType.CONTROL1]
        self.circuit.append(gate(*target).controlled_by(*control, control_values=control_values))

    def finish(self, optimized=False) -> cirq.Circuit:
        if optimized:
            self.circuit = optimize(self.circuit)
        return self.circuit

    def all_qubits(self) -> list:
        return sorted(self.circuit.all_qubits())


def optimize(circuit: Circuit):
    circuit = merge_single_qubit_gates_to_phased_x_and_z(circuit)
    circuit = eject_z(circuit)
    circuit = drop_negligible_operations(circuit)
    circuit = drop_empty_moments(circuit)
    return circuit
