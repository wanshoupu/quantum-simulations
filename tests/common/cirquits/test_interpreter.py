import random

import numpy as np

from common.circuits.cirq_circuit import CirqBuilder
from common.circuits.interpreter import CircuitInterp
from common.construct.quompiler import quompile
from common.utils.mgen import cyclic_matrix, random_unitary

random.seed(42)
np.random.seed(42)


def test_interp_identity_matrix():
    n = random.randint(1, 4)
    dim = 1 << n
    u = np.eye(dim)
    bc = quompile(u)

    builder = CirqBuilder(n)
    interpreter = CircuitInterp(builder)
    interpreter.interpret(bc)
    circuit = builder.finish()
    assert circuit.to_text_diagram() == ''


def test_interp_sing_qubit_circuit():
    n = 1
    dim = 1 << n
    u = random_unitary(dim)
    bc = quompile(u)

    builder = CirqBuilder(n)
    interpreter = CircuitInterp(builder)
    interpreter.interpret(bc)
    circuit = builder.finish()
    # print(circuit)
    qbs = circuit.all_qubits()
    # print(qbs)
    assert len(qbs) == n
    assert len(circuit.moments) == 1


def test_interp_cyclic_matrix():
    n = 3
    dim = 1 << n
    u = cyclic_matrix(dim, 1)
    bc = quompile(u)

    builder = CirqBuilder(n)
    interpreter = CircuitInterp(builder)
    interpreter.interpret(bc)
    circuit = builder.finish()
    # print(circuit)
    qbs = circuit.all_qubits()
    assert len(qbs) == n
    assert len(circuit.moments) == 11


def test_interp_random_unitary():
    for _ in range(1):
        n = random.randint(1, 4)
        dim = 1 << n
        m = random_unitary(dim)
        byte_code = quompile(m)
        builder = CirqBuilder(n)
        interpreter = CircuitInterp(builder)
        interpreter.interpret(byte_code)
        circuit = builder.finish()
        qbs = circuit.all_qubits()
        assert len(qbs) == n
        print(circuit)
