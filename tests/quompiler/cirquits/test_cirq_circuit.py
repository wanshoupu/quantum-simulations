import random

import numpy as np

from quompiler.circuits.cirq_circuit import CirqBuilder
from quompiler.construct.cmat import UnivGate, CUnitary
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_UnitaryM_2l, random_CUnitary, random_indexes, random_control, random_unitary

random.seed(3)
np.random.seed(3)
formatter = MatrixFormatter(precision=2)


def test_create_builder():
    n = 3
    dim = 1 << n
    array = random_UnitaryM_2l(dim, 3, 4)
    cirqC = CirqBuilder(n)
    cirqC.build_gate(array)
    phase = CUnitary(UnivGate.S.mat, (None, False, True))
    print()
    print(formatter.tostr(phase.inflate()))
    cirqC.build_gate(phase)
    circuit = cirqC.finish()
    print(circuit.all_qubits())
    print(circuit)
    u = circuit.unitary(circuit.all_qubits())
    print(formatter.tostr(u))


def test_builder_standard_cunitary():
    for gate in UnivGate:
        print(gate)
        n = random.randint(1, 4)
        control = random_control(n, 1)
        print(f'n={n}, control={control}')
        cu = CUnitary(gate.mat, control)
        cirqC = CirqBuilder(n)

        # execution
        cirqC.build_gate(cu)
        circuit = cirqC.finish()
        print(circuit)
        u = circuit.unitary(sorted(circuit.all_qubits()))
        expected = cu.inflate()
        assert np.allclose(u, expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(u)}'


def test_builder_random_cunitary():
    for _ in range(10):
        print(f'\nTest round #{_} ...')
        n = random.randint(1, 4)
        k = random.randint(1, n)
        control = random_control(n, k)
        core = 1 << control.count(None)
        m = random_unitary(core)
        cu = CUnitary(m, control)
        cirqC = CirqBuilder(n)

        # execution
        cirqC.build_gate(cu)
        circuit = cirqC.finish()
        print(circuit)
        u = circuit.unitary(sorted(circuit.all_qubits()))
        expected = cu.inflate()
        assert np.allclose(u, expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(u)}'
