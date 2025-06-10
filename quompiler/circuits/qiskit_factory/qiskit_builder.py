import warnings
from typing import Union, Sequence, Dict

import numpy as np
from qiskit import QuantumCircuit, AncillaRegister, QuantumRegister
from qiskit.circuit.library import IGate, XGate, HGate, YGate, ZGate, SGate, TGate, SdgGate, TdgGate, RXGate, RYGate, RZGate
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.quantumregister import Qubit as PhysQubit
from typing_extensions import override

from quompiler.circuits.qbuilder import CircuitBuilder
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.qspace import Qubit
from quompiler.construct.types import UnivGate, QType, PrincipalAxis
from quompiler.construct.unitary import UnitaryM


class QiskitBuilder(CircuitBuilder):
    _PRINCIPAL_GATES = {PrincipalAxis.X: RXGate, PrincipalAxis.Y: RYGate, PrincipalAxis.Z: RZGate}

    def __init__(self):
        self.qreg = None
        self.areg = None
        self.circuit = None
        self.qubit_mapping: Dict[Qubit, PhysQubit] = {}

    @override
    def register(self, qspace: Sequence[Qubit]) -> None:
        ancillas = sum(qbit.ancilla for qbit in qspace)
        qubits = len(qspace) - ancillas
        self.areg = AncillaRegister(ancillas)
        self.qreg = QuantumRegister(qubits)
        self.circuit = QuantumCircuit(self.qreg, self.areg)
        qindex, aindex = 0, 0
        for qbit in qspace:
            if qbit.ancilla:
                self.qubit_mapping[qbit] = self.areg[aindex]
                aindex += 1
            else:
                self.qubit_mapping[qbit] = self.qreg[qindex]
                qindex += 1

    @override
    def build_gate(self, m: Union[UnitaryM, CtrlGate]):
        if isinstance(m, CtrlGate):
            m.sorted(np.argsort(m.controls))
            if m.is_std():
                physgate = self.map_gate(m.gate)
            elif m.is_principal():
                principal = m.gate.axis.principal
                angle = m.gate.angle
                physgate = lambda: self._PRINCIPAL_GATES[principal](angle)
            else:
                physgate = lambda: UnitaryGate(m.matrix())
            self._append_gate(physgate, m.qids(), m.controls)
        warnings.warn(f"Warning: gate of type {type(m)} is ignored.")

    def _append_gate(self, gate, qids, controller):
        physical_qubits = [self.qubit_mapping[q] for q in qids]
        control_values = ''.join([str(c.base[0]) for c in controller if c in QType.CONTROL0 | QType.CONTROL1])
        gate_impl = gate()
        if control_values:
            gate_impl = gate_impl.control(len(control_values), ctrl_state=control_values)
        self.circuit.append(gate_impl, physical_qubits)

    @override
    def finish(self, optimized=False) -> QuantumCircuit:
        return self.circuit

    @override
    def all_qubits(self) -> list:
        return [self.qubit_mapping[k] for k in sorted(self.qubit_mapping)]

    def map_gate(self, gate):
        if gate == UnivGate.I:
            return IGate
        if gate == UnivGate.X:
            return XGate
        if gate == UnivGate.Y:
            return YGate
        if gate == UnivGate.Z:
            return ZGate
        if gate == UnivGate.H:
            return HGate
        if gate == UnivGate.T:
            return TGate
        if gate == UnivGate.S:
            return SGate
        if gate == UnivGate.TD:
            return TdgGate
        if gate == UnivGate.SD:
            return SdgGate
        raise Exception(f"Unsupported gate: {gate}")
