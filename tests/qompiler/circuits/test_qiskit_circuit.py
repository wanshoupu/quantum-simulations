import numpy as np
from qiskit.quantum_info import Operator
from typing_extensions import override

from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_UnitaryM_2l
from tests.qompiler.circuits.circuit_test_template import CircuitTestTemplate

from tests.qompiler.mock_fixtures import mock_factory_manager

man = mock_factory_manager(emit="SINGLET", ancilla_offset=100, target="QISKIT")
formatter = MatrixFormatter(precision=2)


def test_create_builder():
    n = 3
    dim = 1 << n
    array = random_UnitaryM_2l(dim, 3, 4)

    factory = man.create_factory()
    qiskitC = factory.get_qompiler()
    assert qiskitC is not None


class TestQiskitCircuit(CircuitTestTemplate):
    man = mock_factory_manager(emit="CTRL_PRUNED", ancilla_offset=100, target="QISKIT")

    @override
    def verify_circuit(self, expected, builder, circuit):
        # print(circuit)
        # print(actual)
        # print(circuit)
        actual = Operator(circuit).data
        # due to Qiskit's weak support for ordering of qubits
        assert np.allclose(actual, expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(actual)}'
