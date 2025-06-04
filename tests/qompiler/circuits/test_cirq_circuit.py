import tempfile

import cirq
import numpy as np
from typing_extensions import override

from quompiler.config.config_manager import ConfigManager
from quompiler.utils.file_io import CODE_FILE_EXT
from quompiler.utils.format_matrix import MatrixFormatter
from tests.qompiler.circuits.circuit_test_template import CircuitTestTemplate

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
    override_config = dict(emit="CTRL_PRUNED", ancilla_offset=100, target="CIRQ")
    config = ConfigManager().merge(override_config).create_config()

    @override
    def verify_circuit(self, expected, builder, circuit):
        # print(circuit)
        # print(actual)
        # print(circuit)
        actual = circuit.unitary(builder.all_qubits())
        assert np.allclose(actual, expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(actual)}'


with tempfile.NamedTemporaryFile(suffix=CODE_FILE_EXT, mode="w+", delete=True) as tmp:
    class TestCirqCircuitEnd2End(CircuitTestTemplate):
        override_config = dict(emit="CTRL_PRUNED", ancilla_offset=100, target="CIRQ")
        config = ConfigManager().merge(override_config).create_config()

        @override
        def verify_circuit(self, expected, builder, circuit):
            # print(circuit)
            # print(circuit)
            actual = circuit.unitary(builder.all_qubits())
            print(f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(actual)}')
            # because this involves Solovay-Kitaev decomposition, we are going to skip verifying the equality of unitary matrices
