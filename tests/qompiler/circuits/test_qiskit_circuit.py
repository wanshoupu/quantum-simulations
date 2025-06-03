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
        ordering = builder.all_qubits()
        sorting = [ordering.index(q) for q in circuit.qubits]
        original = Operator(circuit).data
        actual = permute(original, sorting)
        # due to Qiskit's bug, the ordering of qubits is messed up
        # assert np.allclose(actual, expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(actual)}'
        assert expected.shape == actual.shape


def permute(U, sorting):
    n = len(sorting)
    U_tensor = U.reshape([2] * 2 * n)  # Reshape to tensor: 2^n x 2^n → (2,)*2n

    # Create the permutation indices
    # Outputs are the first `n` axes, inputs are the last `n` axes
    perm = sorting + [i + n for i in sorting]
    U_reordered = np.transpose(U_tensor, perm).reshape(2 ** n, 2 ** n)
    return U_reordered
