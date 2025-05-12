import random
from functools import reduce

import numpy as np

from quompiler.construct.bytecode import BytecodeIter, Bytecode
from quompiler.construct.cgate import CtrlGate
from quompiler.qompile.qompiler import Qompiler
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import cyclic_matrix, random_unitary
from tests.qompiler.qompile.mock_fixtures import mock_config

formatter = MatrixFormatter(precision=2)


def test_compile_identity_matrix(mocker):
    n = 3
    dim = 1 << n
    u = np.eye(dim)
    config = mock_config(mocker)
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
    config = mock_config(mocker, emit="SINGLET")
    interp = Qompiler(config)

    # execute
    bc = interp.compile(u)
    # print(bc)
    assert isinstance(bc, Bytecode)
    assert len(bc.children) == 1
    data = bc.children[0].data
    assert isinstance(data, CtrlGate)


def test_compile_cyclic_8(mocker):
    u = cyclic_matrix(8, 1)
    config = mock_config(mocker, emit="SINGLET")
    interp = Qompiler(config)

    # execute
    bc = interp.compile(u)
    # print(bc)
    assert bc is not None
    data = [a.data for a in BytecodeIter(bc)]
    assert len(data) == 21
    leaves = [a.data.inflate() for a in BytecodeIter(bc) if a.is_leaf()]
    v = reduce(lambda a, b: a @ b, leaves)
    assert np.allclose(v, u), f'circuit != input:\ncompiled=\n{formatter.tostr(v)},\ninput=\n{formatter.tostr(u)}'


def test_compile_cyclic_4(mocker):
    u = cyclic_matrix(4, 1)
    config = mock_config(mocker, emit="SINGLET")
    interp = Qompiler(config)

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


def test_interp_random_unitary(mocker):
    for _ in range(10):
        # print(f'Test {_}th round')
        n = random.randint(1, 4)
        dim = 1 << n
        u = random_unitary(dim)
        config = mock_config(mocker, emit="SINGLET")
        interp = Qompiler(config)

        # execute
        bc = interp.compile(u)

        # verify
        leaves = [a.data.inflate() for a in BytecodeIter(bc) if isinstance(a.data, CtrlGate)]
        v = reduce(lambda a, b: a @ b, leaves)
        assert np.allclose(v, u), f'circuit != input:\ncompiled=\n{formatter.tostr(v)},\ninput=\n{formatter.tostr(u)}'
