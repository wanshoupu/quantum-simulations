import random
from functools import reduce

from quompiler.construct.bytecode import BytecodeIter
from quompiler.construct.cgate import CtrlGate
from quompiler.qompile.configure import DeviceConfig, QompilerConfig
from quompiler.qompile.quompiler import Qompiler
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import cyclic_matrix, random_unitary
import numpy as np

from tests.quompiler.qompile.mock_fixtures import mock_config

formatter = MatrixFormatter(precision=2)


def test_compile_identity_matrix(mocker):
    n = 3
    dim = 1 << n
    u = np.eye(dim)
    config = mock_config(mocker, dim)
    interp = Qompiler(config)

    # execute
    bc = interp.compile(u)
    assert bc is not None
    assert np.array_equal(bc.data.matrix, np.eye(bc.data.matrix.shape[0]))
    assert bc.children == []


def test_compile_sing_qubit_circuit(mocker):
    n = 1
    dim = 1 << n
    u = random_unitary(dim)
    config = mock_config(mocker, dim)
    interp = Qompiler(config)

    # execute
    bc = interp.compile(u)
    # print(bc)
    assert bc is not None
    assert 1 == len([a for a in BytecodeIter(bc)])
    assert isinstance(bc.data, CtrlGate)


def test_compile_cyclic_8():
    u = cyclic_matrix(8, 1)
    device = DeviceConfig(dimension=8)
    config = QompilerConfig(source='foo', device=device)
    interp = Qompiler(config)

    # execute
    bc = interp.compile(u)
    # print(bc)
    assert bc is not None
    assert 18 == len([a for a in BytecodeIter(bc)])
    leaves = [a.data.inflate() for a in BytecodeIter(bc) if isinstance(a.data, CtrlGate)]
    v = reduce(lambda a, b: a @ b, leaves)
    assert np.allclose(v, u), f'circuit != input:\ncompiled=\n{formatter.tostr(v)},\ninput=\n{formatter.tostr(u)}'


def test_compile_cyclic_4():
    u = cyclic_matrix(4, 1)
    device = DeviceConfig(dimension=4)
    config = QompilerConfig(source='foo', device=device)
    interp = Qompiler(config)

    # execute
    bc = interp.compile(u)
    # print(bc)
    assert bc is not None
    assert 6 == len([a for a in BytecodeIter(bc)])
    leaves = [a.data.inflate() for a in BytecodeIter(bc) if isinstance(a.data, CtrlGate)]
    assert len(leaves) == 4
    v = reduce(lambda a, b: a @ b, leaves)
    assert np.allclose(v, u), f'circuit != input:\ncompiled=\n{formatter.tostr(v)},\ninput=\n{formatter.tostr(u)}'


def test_interp_random_unitary():
    for _ in range(10):
        # print(f'Test {_}th round')
        n = random.randint(1, 4)
        dim = 1 << n
        u = random_unitary(dim)
        device = DeviceConfig(dimension=dim)
        config = QompilerConfig(source='foo', device=device)
        interp = Qompiler(config)

        # execute
        bc = interp.compile(u)
        leaves = [a.data.inflate() for a in BytecodeIter(bc) if isinstance(a.data, CtrlGate)]
        v = reduce(lambda a, b: a @ b, leaves)
        assert np.allclose(v, u), f'circuit != input:\ncompiled=\n{formatter.tostr(v)},\ninput=\n{formatter.tostr(u)}'
