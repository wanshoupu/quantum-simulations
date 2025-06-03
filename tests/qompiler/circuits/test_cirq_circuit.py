import cirq
import numpy as np
from typing_extensions import override

from quompiler.utils.format_matrix import MatrixFormatter
from tests.qompiler.circuits.circuit_test_template import CircuitTestTemplate
from tests.qompiler.mock_fixtures import mock_factory_manager

formatter = MatrixFormatter(precision=2)

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
    man = mock_factory_manager(emit="CLIFFORD_T", ancilla_offset=100, target="CIRQ")

    @override
    def verify_circuit(self, expected, builder, circuit):
        # print(circuit)
        # print(actual)
        # print(circuit)
        actual = circuit.unitary(builder.all_qubits())
        assert np.allclose(actual, expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(actual)}'
