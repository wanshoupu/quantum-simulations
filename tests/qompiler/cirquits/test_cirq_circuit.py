import random

import cirq
import numpy as np

from quompiler.construct.bytecode import BytecodeIter
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.types import UnivGate, QType
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_control, random_unitary, cyclic_matrix
from tests.qompiler.mock_fixtures import mock_factory_manager

formatter = MatrixFormatter(precision=2)

man = mock_factory_manager(emit="SINGLET", ancilla_offset=100)


def test_create_builder(mocker):
    factory = man.create_factory()
    cirqC = factory.get_builder()
    phase = CtrlGate(UnivGate.S.matrix, (QType.TARGET, QType.CONTROL0, QType.CONTROL1))
    # print()
    # print(formatter.tostr(phase.inflate()))
    cirqC.build_gate(phase)
    circuit = cirqC.finish()
    # print(circuit.all_qubits())
    # print(circuit)
    u = circuit.unitary(circuit.all_qubits())
    print(formatter.tostr(u))


def test_builder_standard_ctrlgate(mocker):
    for gate in UnivGate:
        # print(gate)
        n = random.randint(1, 4)
        control = random_control(n, 1)
        # print(f'n={n}, control={control}')
        cu = CtrlGate(gate.matrix, control)
        man = mock_factory_manager(emit="SINGLET", ancilla_offset=100)
        factory = man.create_factory()
        cirqC = factory.get_builder()

        # execution
        cirqC.build_gate(cu)
        circuit = cirqC.finish()
        # print(circuit)
        u = circuit.unitary(sorted(circuit.all_qubits()))
        expected = cu.inflate()
        assert np.allclose(u, expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(u)}'


def test_builder_random_ctrlgate(mocker):
    for _ in range(10):
        # print(f'\nTest round #{_} ...')
        n = random.randint(1, 4)
        k = random.randint(1, n)
        control = random_control(n, k)
        core = 1 << control.count(QType.TARGET)
        m = random_unitary(core)
        cu = CtrlGate(m, control)
        factory = man.create_factory()
        cirqC = factory.get_builder()

        # execution
        cirqC.build_gate(cu)
        circuit = cirqC.finish()
        # print(circuit)
        u = circuit.unitary(sorted(circuit.all_qubits()))
        expected = cu.inflate()
        assert np.allclose(u, expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(u)}'


def test_compile_cyclic_4_everything(mocker):
    n = 2
    dim = 1 << n
    u = cyclic_matrix(dim, 1)
    factory = man.create_factory()
    interp = factory.get_qompiler()

    # execute
    bc = interp.compile(u)
    # print(bc)
    assert len([a for a in BytecodeIter(bc)]) == 7
    # we need to revert the order bc the last element appears first in the circuit
    leaves = [a.data for a in BytecodeIter(bc) if isinstance(a.data, CtrlGate)][::-1]
    assert len(leaves) == 4

    cirqC = factory.get_builder()

    # execution
    for cu in leaves:
        cirqC.build_gate(cu)
    circuit = cirqC.finish()
    # print(circuit)
    v = circuit.unitary(cirqC.all_qubits())
    assert np.allclose(v, u), f'circuit != input:\ncompiled=\n{formatter.tostr(v)},\ninput=\n{formatter.tostr(u)}'


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
