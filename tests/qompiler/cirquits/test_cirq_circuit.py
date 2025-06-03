import cirq
import numpy as np

from tests.qompiler.cirquits.circuit_test_template import CircuitTestTemplate
from tests.qompiler.mock_fixtures import mock_factory_manager


def test_cirq_bug_4_qubits():
    n = 4
    qubits = cirq.LineQubit.range(n)
    circuit = cirq.Circuit()
    custom_gate = cirq.MatrixGate(np.eye(1 << n))

    # execute
    circuit.append(custom_gate(*qubits))

    assert qubits != circuit.all_qubits()

    # to bypass the cirq bug, always sort circuit.all_qubits().
    assert qubits == sorted(circuit.all_qubits())


class TestCirqCircuit(CircuitTestTemplate):
    man = mock_factory_manager(emit="SINGLET", ancilla_offset=100, target="CIRQ")
