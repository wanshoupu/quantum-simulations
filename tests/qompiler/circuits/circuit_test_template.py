import random

import numpy as np
import pytest

from quompiler.circuits.factory_manager import FactoryManager
from quompiler.construct.bytecode import BytecodeIter
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.types import UnivGate, QType
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_control, random_unitary, cyclic_matrix

formatter = MatrixFormatter(precision=2)


class CircuitTestTemplate:
    man: FactoryManager = None

    def test_create_builder(self):
        factory = self.man.create_factory()
        builder = factory.get_builder()

        phase = CtrlGate(np.array(UnivGate.S), (QType.TARGET, QType.CONTROL0, QType.CONTROL1))
        # print()
        # print(formatter.tostr(phase.inflate()))
        builder.register(phase.qspace)
        builder.build_gate(phase)
        circuit = builder.finish()
        # print(circuit.all_qubits())
        # print(circuit)
        u = circuit.unitary(circuit.all_qubits())
        print(formatter.tostr(u))

    @pytest.mark.parametrize("gate", list(UnivGate))
    def test_builder_standard_ctrlgate(self, gate):
        factory = self.man.create_factory()
        builder = factory.get_builder()

        # print(gate)
        n = random.randint(1, 4)
        control = random_control(n, 1)
        # print(f'n={n}, control={control}')
        cu = CtrlGate(np.array(gate), control)

        # execution
        builder.register(cu.qspace)
        builder.build_gate(cu)
        circuit = builder.finish()
        # print(circuit)
        u = circuit.unitary(sorted(circuit.all_qubits()))
        expected = cu.inflate()
        assert np.allclose(u, expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(u)}'

    @pytest.mark.parametrize("seed", random.sample(range(1 << 20), 10))
    def test_builder_random_ctrlgate(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)

        factory = self.man.create_factory()
        builder = factory.get_builder()

        # print(f'\nTest round #{_} ...')
        n = random.randint(1, 4)
        k = random.randint(1, n)
        control = random_control(n, k)
        core = 1 << control.count(QType.TARGET)
        m = random_unitary(core)
        cu = CtrlGate(m, control)

        # execution
        builder.register(cu.qspace)
        builder.build_gate(cu)
        circuit = builder.finish()
        # print(circuit)
        u = circuit.unitary(sorted(circuit.all_qubits()))
        expected = cu.inflate()
        assert np.allclose(u, expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(u)}'

    def test_builder_cyclic_4_everything(self):
        n = 2
        dim = 1 << n
        u = cyclic_matrix(dim, 1)

        factory = self.man.create_factory()
        builder = factory.get_builder()
        compiler = factory.get_qompiler()

        # execute
        bc = compiler.decompose(u)
        # print(bc)
        assert len([a for a in BytecodeIter(bc)]) == 7
        # we need to revert the order bc the last element appears first in the circuit
        leaves = [a.data for a in BytecodeIter(bc) if isinstance(a.data, CtrlGate)][::-1]
        assert len(leaves) == 4

        # execution
        for cu in leaves:
            builder.register(cu.qspace)
            builder.build_gate(cu)
        circuit = builder.finish()
        # print(circuit)
        v = circuit.unitary(builder.all_qubits())
        assert np.allclose(v, u), f'circuit != input:\ncompiled=\n{formatter.tostr(v)},\ninput=\n{formatter.tostr(u)}'
