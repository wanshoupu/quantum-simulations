import random
from abc import abstractmethod, ABC

import numpy as np
import pytest

from quompiler.circuits.qfactory import QFactory
from quompiler.config.construct import QompilerConfig
from quompiler.construct.bytecode import BytecodeIter
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.types import UnivGate, QType, QompilePlatform
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_control, random_unitary, cyclic_matrix

formatter = MatrixFormatter(precision=2)


class CircuitTestTemplate(ABC):
    config: QompilerConfig = None

    def test_create_builder(self):
        factory = QFactory(self.config)
        builder = factory.get_builder(self.config.target)

        phase = CtrlGate(np.array(UnivGate.S), (QType.TARGET, QType.CONTROL0, QType.CONTROL1))
        # print()
        # print(formatter.tostr(phase.inflate()))
        builder.register(phase.qspace)
        builder.build_gate(phase)
        circuit = builder.finish()
        assert circuit is not None
        self.verify_circuit(phase.inflate(), builder, circuit)

    @pytest.mark.parametrize("gate", list(UnivGate))
    def test_builder_standard_ctrlgate(self, gate):
        factory = QFactory(self.config)
        builder = factory.get_builder(self.config.target)

        # print(gate)
        n = random.randint(1, 4)
        control = random_control(n, 1)
        # print(f'n={n}, control={control}')
        cu = CtrlGate(np.array(gate), control)

        # execution
        builder.register(cu.qspace)
        builder.build_gate(cu)
        circuit = builder.finish()
        assert circuit is not None
        self.verify_circuit(cu.inflate(), builder, circuit)

    @pytest.mark.parametrize("seed", random.sample(range(1 << 20), 10))
    def test_builder_random_ctrlgate(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)

        factory = QFactory(self.config)
        builder = factory.get_builder(self.config.target)

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
        assert circuit is not None
        self.verify_circuit(cu.inflate(), builder, circuit)

    def test_builder_cyclic_4_everything(self):
        n = 2
        dim = 1 << n
        u = cyclic_matrix(dim, 1)

        factory = QFactory(self.config)
        builder = factory.get_builder(self.config.target)
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
        assert circuit is not None
        self.verify_circuit(u, builder, circuit)

    def test_builder_random_end_2_end(self):
        n = 1
        dim = 1 << n
        u = random_unitary(dim)
        factory = QFactory(self.config)
        interp = factory.get_qompiler()

        # execute
        interp.compile(u)
        render = factory.get_render(self.config.target)
        codefile = factory.get_config().output
        circuit = render.render(codefile)
        assert circuit is not None
        self.verify_circuit(u, render.builder, circuit)

    @abstractmethod
    def verify_circuit(self, unitary, builder, circuit):
        pass
