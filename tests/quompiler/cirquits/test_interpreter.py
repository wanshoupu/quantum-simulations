import random

import numpy as np
import pytest

from quompiler.circuits.cirq_circuit import CirqBuilder
from quompiler.qompile.configure import CompilerConfig, DeviceConfig
from quompiler.qompile.quompiler import CircuitInterp
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import cyclic_matrix, random_unitary

formatter = MatrixFormatter(precision=2)


def test_interp_identity_matrix():
    n = random.randint(1, 8)
    dim = 1 << n
    expected = np.eye(dim)

    device = DeviceConfig(dimension=dim)
    config = CompilerConfig(source='foo', device=device)
    interp = CircuitInterp(config)
    interp.interpret(expected)
    circuit = interp.finish()
    qubits = circuit.all_qubits()
    actual = circuit.unitary(qubits)
    assert np.allclose(actual, 1), f'circuit != input:\ncircuit=\n{actual},\ninput=\n{expected}'
    assert circuit.to_text_diagram() == ''


def test_interp_sing_qubit_circuit():
    n = 1
    dim = 1 << n
    expected = random_unitary(dim)

    device = DeviceConfig(dimension=dim)
    config = CompilerConfig(source='foo', device=device)
    interp = CircuitInterp(config)

    # execute
    interp.interpret(expected)
    circuit = interp.finish()

    # verify
    qbs = circuit.all_qubits()
    actual = circuit.unitary(qbs)
    assert np.allclose(actual, expected), f'actual != expected:\nactual=\n{actual},\nexpected=\n{expected}'
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
    expected = cyclic_matrix(dim, k)

    device = DeviceConfig(dimension=dim)
    config = CompilerConfig(source='foo', device=device)
    interp = CircuitInterp(config)

    # execute
    interp.interpret(expected)
    circuit = interp.finish()

    # verify
    qbs = interp.all_qubits()
    actual = circuit.unitary(qbs)
    assert len(qbs) == n
    assert np.allclose(actual, expected), f'actual != expected:\nactual=\n{actual},\nexpected=\n{expected}'
    assert len(circuit.moments) == expected_moments


def test_interp_random_unitary():
    for _ in range(10):
        print(f'Test {_}th round')
        n = random.randint(1, 4)
        dim = 1 << n
        expected = random_unitary(dim)

        device = DeviceConfig(dimension=dim)
        config = CompilerConfig(source='foo', device=device)
        interp = CircuitInterp(config)

        # execute
        interp.interpret(expected)
        circuit = interp.finish()

        # verify
        qbs = interp.all_qubits()
        actual = circuit.unitary(qbs)
        assert len(qbs) == n
        assert np.allclose(actual, expected), f'actual != expected:\nactual=\n{formatter.tostr(actual)},\nexpected=\n{formatter.tostr(expected)}'
