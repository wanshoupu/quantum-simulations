import random

import numpy as np
import pytest

from quompiler.circuits.cirq_circuit import CirqBuilder
from quompiler.qompile.quompiler import CircuitInterp
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import cyclic_matrix, random_unitary

formatter = MatrixFormatter(precision=2)


def test_interp_identity_matrix():
    n = random.randint(1, 8)
    dim = 1 << n
    u = np.eye(dim)

    builder = CirqBuilder(n)
    CircuitInterp(builder).interpret(u)
    circuit = builder.finish()
    c = circuit.unitary(circuit.all_qubits())
    assert np.allclose(c, 1), f'circuit != input:\ncircuit=\n{c},\ninput=\n{u}'
    assert circuit.to_text_diagram() == ''


def test_interp_sing_qubit_circuit():
    n = 1
    dim = 1 << n
    u = random_unitary(dim)

    builder = CirqBuilder(n)
    CircuitInterp(builder).interpret(u)
    circuit = builder.finish()
    # print()
    # print(formatter.tostr(c))
    c = circuit.unitary(circuit.all_qubits())
    assert np.allclose(u, c), f'circuit != input:\ncircuit=\n{c},\ninput=\n{u}'
    qbs = circuit.all_qubits()
    # print(qbs)
    assert len(qbs) == n
    assert len(circuit.moments) == 1


@pytest.mark.parametrize("n,k,expected_moments", [
    (2, 1, 4),
    (3, 1, 14),
    (3, 2, 11),
])
def test_interp_cyclic_matrix(n, k, expected_moments):
    dim = 1 << n
    u = cyclic_matrix(dim, k)

    builder = CirqBuilder(n)
    CircuitInterp(builder).interpret(u)
    circuit = builder.finish()

    qbs = sorted(circuit.all_qubits())
    assert len(qbs) == n

    c = circuit.unitary(qbs)
    assert np.allclose(u, c), f'circuit != input:\ncircuit=\n{formatter.tostr(c)},\ninput=\n{formatter.tostr(u)}'
    assert len(circuit.moments) == expected_moments


def test_interp_random_unitary():
    for _ in range(10):
        print(f'Test {_}th round')
        n = random.randint(1, 4)
        dim = 1 << n
        expected = random_unitary(dim)

        # execute
        builder = CirqBuilder(n)
        CircuitInterp(builder).interpret(expected)
        circuit = builder.finish()
        qbs = circuit.all_qubits()
        assert len(qbs) == n
        actual = circuit.unitary(builder.qubits)
        assert np.allclose(actual, expected), f'actual != expected:\nactual=\n{formatter.tostr(actual)},\nexpected=\n{formatter.tostr(expected)}'
