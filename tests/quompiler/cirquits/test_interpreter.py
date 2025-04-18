import random

import numpy as np

from quompiler.circuits.cirq_circuit import CirqBuilder
from quompiler.circuits.interpreter import CircuitInterp
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import cyclic_matrix, random_unitary

random.seed(42)
np.random.seed(42)
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


def test_interp_cyclic_matrix():
    n = 2
    dim = 1 << n
    u = cyclic_matrix(dim, 1)

    builder = CirqBuilder(n)
    CircuitInterp(builder).interpret(u)
    circuit = builder.finish()
    # print(circuit)
    c = circuit.unitary()
    assert np.allclose(u, c), f'circuit != input:\ncircuit=\n{formatter.tostr(c)},\ninput=\n{formatter.tostr(u)}'
    qbs = circuit.all_qubits()
    assert len(qbs) == n
    assert len(circuit.moments) == 14


def test_interp_random_unitary():
    for _ in range(3):
        print(f'Test {_}th round')
        n = random.randint(1, 4)
        dim = 1 << n
        m = random_unitary(dim)
        builder = CirqBuilder(n)
        CircuitInterp(builder).interpret(m)
        circuit = builder.finish()
        qbs = circuit.all_qubits()
        assert len(qbs) == n
        c = circuit.unitary(circuit.all_qubits())
        assert np.allclose(m, c), f'circuit != input:\ncircuit=\n{formatter.tostr(c)},\ninput=\n{formatter.tostr(m)}'
