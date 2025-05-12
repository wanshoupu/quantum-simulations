from typing import Optional, Union

import cirq
from cirq import EigenGate, Circuit, merge_single_qubit_gates_to_phased_x_and_z, eject_z, drop_negligible_operations, drop_empty_moments
from typing_extensions import override

from quompiler.circuits.circuit_builder import CircuitBuilder
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.std_gate import CtrlStdGate
from quompiler.construct.unitary import UnitaryM
from quompiler.construct.types import UnivGate, QType
from quompiler.qompile.configure import DeviceConfig


def optimize(circuit: Circuit):
    circuit = merge_single_qubit_gates_to_phased_x_and_z(circuit)
    circuit = eject_z(circuit)
    circuit = drop_negligible_operations(circuit)
    circuit = drop_empty_moments(circuit)
    return circuit


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

    @override
    def __init__(self, deviceConfig: DeviceConfig):
        a, b = deviceConfig.arange

        self.max_qid = a
        self.qspace = cirq.LineQubit.range(a)
        self.aspace = cirq.LineQubit.range(a, b)

        self.circuit = cirq.Circuit()

    @override
    def get_univ_gate(self, m: Union[UnitaryM, CtrlGate, CtrlStdGate]) -> Optional[EigenGate]:
        if isinstance(m, CtrlGate) or isinstance(m, CtrlStdGate):
            matrix = m.gate.matrix if isinstance(m, CtrlStdGate) else m.unitary.matrix
            univ_gate = UnivGate.get(matrix)
            if univ_gate:
                return CirqBuilder.__UNIV_GATES[univ_gate]
        return None

    @override
    def build_gate(self, m: Union[UnitaryM, CtrlGate, CtrlStdGate]):
        if isinstance(m, CtrlGate):
            gate = self.get_univ_gate(m) or cirq.MatrixGate(m.unitary.matrix)
            qids = m.qids()
            if self.max_qid <= len(qids):
                raise MemoryError("not enough qubits")
            controller = m.controller
            self._append_gate(controller, gate, qids)
            return

    def _append_gate(self, controller, gate, qids):
        target = [self.qspace[qids[i]] for i, c in enumerate(controller) if c is QType.TARGET]
        control = [self.qspace[qids[i]] for i, c in enumerate(controller) if c in QType.CONTROL0 | QType.CONTROL1]
        control_values = [c.base[0] for c in controller if c in QType.CONTROL0 | QType.CONTROL1]
        self.circuit.append(gate(*target).controlled_by(*control, control_values=control_values))

    @override
    def finish(self, optimized=False) -> cirq.Circuit:
        if optimized:
            self.circuit = optimize(self.circuit)
        return self.circuit

    @override
    def all_qubits(self) -> list:
        return sorted(self.circuit.all_qubits())
