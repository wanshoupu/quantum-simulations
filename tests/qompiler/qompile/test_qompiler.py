import random
from functools import reduce

import numpy as np

from quompiler.construct.bytecode import BytecodeIter, Bytecode
from quompiler.construct.cgate import CtrlGate
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import cyclic_matrix, random_unitary

formatter = MatrixFormatter(precision=2)


def test_compile_identity_matrix():
    from tests.qompiler.mock_fixtures import mock_factory_manager
    man = mock_factory_manager(emit="SINGLET")
    n = 3
    dim = 1 << n
    u = np.eye(dim)
    factory = man.create_factory()
    interp = factory.get_qompiler()

    # execute
    bc = interp.compile(u)
    assert bc is not None
    assert np.array_equal(bc.data.matrix, np.eye(bc.data.matrix.shape[0]))
    assert bc.children == []


def test_compile_sing_qubit_circuit():
    from tests.qompiler.mock_fixtures import mock_factory_manager
    man = mock_factory_manager(emit="SINGLET")
    n = 1
    dim = 1 << n
    u = random_unitary(dim)
    factory = man.create_factory()
    interp = factory.get_qompiler()

    # execute
    bc = interp.compile(u)
    # print(bc)
    assert isinstance(bc, Bytecode)
    assert len(bc.children) == 1
    data = bc.children[0].data
    assert isinstance(data, CtrlGate)


def test_compile_cyclic_8_ctrl_prune():
    from tests.qompiler.mock_fixtures import mock_factory_manager
    u = cyclic_matrix(8, 1)
    man = mock_factory_manager(emit="CTRL_PRUNED")
    factory = man.create_factory()
    interp = factory.get_qompiler()

    # execute
    bc = interp.compile(u)
    # print(bc)
    assert bc is not None
    data = [a.data for a in BytecodeIter(bc)]
    assert len(data) == 559
    leaves = [a.data for a in BytecodeIter(bc) if a.is_leaf()]
    v = reduce(lambda a, b: a @ b, leaves).dela().inflate()
    assert np.allclose(v, u), f'circuit != input:\ncompiled=\n{formatter.tostr(v)},\ninput=\n{formatter.tostr(u)}'


def test_compile_cyclic_8():
    from tests.qompiler.mock_fixtures import mock_factory_manager
    man = mock_factory_manager(emit="SINGLET")
    u = cyclic_matrix(8, 1)
    factory = man.create_factory()
    interp = factory.get_qompiler()

    # execute
    bc = interp.compile(u)
    # print(bc)
    assert bc is not None
    data = [a.data for a in BytecodeIter(bc)]
    assert len(data) == 21
    leaves = [a.data.inflate() for a in BytecodeIter(bc) if a.is_leaf()]
    v = reduce(lambda a, b: a @ b, leaves)
    assert np.allclose(v, u), f'circuit != input:\ncompiled=\n{formatter.tostr(v)},\ninput=\n{formatter.tostr(u)}'


def test_compile_cyclic_4():
    from tests.qompiler.mock_fixtures import mock_factory_manager
    man = mock_factory_manager(emit="SINGLET")
    u = cyclic_matrix(4, 1)
    factory = man.create_factory()
    interp = factory.get_qompiler()

    # execute
    bc = interp.compile(u)
    # print(bc)
    assert bc is not None
    data = [a.data for a in BytecodeIter(bc)]
    assert len(data) == 7
    leaves = [a.data.inflate() for a in BytecodeIter(bc) if a.is_leaf()]
    assert len(leaves) == 4
    v = reduce(lambda a, b: a @ b, leaves)
    assert np.allclose(v, u), f'circuit != input:\ncompiled=\n{formatter.tostr(v)},\ninput=\n{formatter.tostr(u)}'


def test_interp_random_unitary():
    from tests.qompiler.mock_fixtures import mock_factory_manager
    man = mock_factory_manager(emit="SINGLET")
    for _ in range(10):
        # print(f'Test {_}th round')
        n = random.randint(1, 4)
        dim = 1 << n
        u = random_unitary(dim)
        factory = man.create_factory()
        interp = factory.get_qompiler()

        # execute
        bc = interp.compile(u)

        # verify
        leaves = [a.data.inflate() for a in BytecodeIter(bc) if isinstance(a.data, CtrlGate)]
        v = reduce(lambda a, b: a @ b, leaves)
        assert np.allclose(v, u), f'circuit != input:\ncompiled=\n{formatter.tostr(v)},\ninput=\n{formatter.tostr(u)}'


def test_optimize_no_optimizer():
    from tests.qompiler.mock_fixtures import mock_factory_manager
    from quompiler.circuits.qompiler import Qompiler
    man = mock_factory_manager(emit="SINGLET")
    cman = man.config_man
    factory = man.create_factory()
    compiler = Qompiler(cman.create_config(), factory.get_builder(), factory.get_device())
    u = random_unitary(2)
    code = compiler.compile(u)

    # execute
    optcode = compiler.optimize(code)
    assert optcode is not None
    leaves = [a.data for a in BytecodeIter(optcode) if a.is_leaf()]
    v = reduce(lambda a, b: a @ b, leaves)
    assert np.allclose(v.inflate(), u), f'circuit != input:\ncompiled=\n{formatter.tostr(v.inflate())},\ninput=\n{formatter.tostr(u)}'


def test_optimize_basic_optimizer():
    from tests.qompiler.mock_fixtures import mock_factory_manager
    man = mock_factory_manager(emit="SINGLET")
    factory = man.create_factory()
    compiler = factory.get_qompiler()
    u = random_unitary(2)
    code = compiler.compile(u)

    # execute
    optcode = compiler.optimize(code)
    assert optcode is not None
    leaves = [a.data for a in BytecodeIter(optcode) if a.is_leaf()]
    v = reduce(lambda a, b: a @ b, leaves)
    assert np.allclose(v.inflate(), u), f'circuit != input:\ncompiled=\n{formatter.tostr(v.inflate())},\ninput=\n{formatter.tostr(u)}'
