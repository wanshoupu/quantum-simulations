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
        self.qspace = cirq.LineQubit.range(*deviceConfig.qrange)
        self.aspace = cirq.LineQubit.range(*deviceConfig.arange)

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
            # TODO add gate approximation
            gate = self.get_univ_gate(m) or cirq.MatrixGate(m.unitary.matrix)
            target = [self.qubits[i] for i, c in enumerate(m.controller.controls) if c is QType.TARGET]
            control = [self.qubits[i] for i, c in enumerate(m.controller.controls) if c in QType.CONTROL0 | QType.CONTROL1]
            control_values = [c.base[0] for c in m.controller.controls if c in QType.CONTROL0 | QType.CONTROL1]
            self.circuit.append(gate(*target).controlled_by(*control, control_values=control_values))
            return

        # TODO add KronUnitaryM gate
        custom_gate = cirq.MatrixGate(m.inflate())
        self.circuit.append(custom_gate(*self.qubits))

    @override
    def finish(self, optimized=False) -> cirq.Circuit:
        if optimized:
            self.circuit = optimize(self.circuit)
        return self.circuit

    @override
    def all_qubits(self) -> list:
        return sorted(self.circuit.all_qubits())
