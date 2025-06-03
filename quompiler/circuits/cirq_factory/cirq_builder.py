import warnings
from typing import Union, Optional, Dict

import cirq
from cirq import EigenGate, Circuit, merge_single_qubit_gates_to_phased_x_and_z, eject_z, drop_negligible_operations, drop_empty_moments
from typing_extensions import override

from quompiler.circuits.qbuilder import CircuitBuilder
from quompiler.config.construct import QompilerConfig
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.qspace import Qubit
from quompiler.construct.types import UnivGate, QType
from quompiler.construct.unitary import UnitaryM


class CirqBuilder(CircuitBuilder):
    _UNIV_GATES = {
        # identity gate for completeness
        UnivGate.I: cirq.I,
        # necessary gates
        UnivGate.H: cirq.H,
        UnivGate.S: cirq.S,
        UnivGate.T: cirq.T,
        UnivGate.SD: lambda q: cirq.S(q) ** -1,
        UnivGate.TD: lambda q: cirq.T(q) ** -1,
        # additional auxiliary gates
        UnivGate.X: cirq.X,
        UnivGate.Y: cirq.Y,
        UnivGate.Z: cirq.Z,
    }

    def __init__(self):
        self.circuit = cirq.Circuit()
        self.qubit_mapping: Dict[Qubit, object] = {}

    def get_univ_gate(self, gate: UnivGate) -> Optional[EigenGate]:
        return self._UNIV_GATES[gate]

    def build_gate(self, m: Union[UnitaryM, CtrlGate]):
        if isinstance(m, CtrlGate):
            gate = self.get_univ_gate(m.gate) if m.is_std() else cirq.MatrixGate(m.matrix())
            self._append_gate(m.controls, gate, m.qids())
        warnings.warn(f"Warning: gate of type {type(m)} is ignored.")

    def _append_gate(self, controller, gate, qids):
        target = [self.qubit_mapping[qids[i]] for i, c in enumerate(controller) if c is QType.TARGET]
        control = [self.qubit_mapping[qids[i]] for i, c in enumerate(controller) if c in QType.CONTROL0 | QType.CONTROL1]
        control_values = [c.base[0] for c in controller if c in QType.CONTROL0 | QType.CONTROL1]
        self.circuit.append(gate(*target).controlled_by(*control, control_values=control_values))

    def finish(self, optimized=False) -> cirq.Circuit:
        if optimized:
            self.circuit = optimize(self.circuit)
        return self.circuit

    def all_qubits(self) -> list:
        return [self.qubit_mapping[k] for k in sorted(self.qubit_mapping)]

    @override
    def register(self, qspace):
        for qbit in qspace:
            if qbit in self.qubit_mapping:
                continue
            self.qubit_mapping[qbit] = cirq.NamedQubit(str(qbit))


def optimize(circuit: Circuit):
    circuit = merge_single_qubit_gates_to_phased_x_and_z(circuit)
    circuit = eject_z(circuit)
    circuit = drop_negligible_operations(circuit)
    circuit = drop_empty_moments(circuit)
    return circuit
