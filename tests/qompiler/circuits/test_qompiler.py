import os
import random
import tempfile
from functools import reduce

import numpy as np
import pytest

from quompiler.circuits.qfactory import QFactory
from quompiler.config.config_manager import ConfigManager
from quompiler.construct.bytecode import BytecodeIter, Bytecode
from quompiler.construct.cgate import CtrlGate
from quompiler.utils.file_io import CODE_FILE_EXT
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import cyclic_matrix, random_unitary

formatter = MatrixFormatter(precision=2)


def test_compile_identity_matrix():
    n = 3
    dim = 1 << n
    u = np.eye(dim)
    override = dict(emit="SINGLET", ancilla_offset=n, target="QISKIT")
    config = ConfigManager().merge(override).create_config()
    factory = QFactory(config)
    interp = factory.get_qompiler()

    # execute
    bc = interp.decompose(u)
    assert bc is not None
    assert np.array_equal(bc.data.matrix, np.eye(bc.data.matrix.shape[0]))
    assert bc.children == []


def test_compile_sing_qubit_circuit():
    n = 1
    dim = 1 << n
    u = random_unitary(dim)
    override = dict(emit="SINGLET", ancilla_offset=n)
    config = ConfigManager().merge(override).create_config()
    factory = QFactory(config)
    interp = factory.get_qompiler()

    # execute
    bc = interp.decompose(u)
    # print(bc)
    assert isinstance(bc, Bytecode)
    assert len(bc.children) == 1
    data = bc.children[0].data
    assert isinstance(data, CtrlGate)


def test_compile_insufficient_qspace_error():
    # TODO: ancilla_offset=1 is not working
    override = dict(emit="CTRL_PRUNED", ancilla_offset=1)
    config = ConfigManager().merge(override).create_config()
    factory = QFactory(config)
    interp = factory.get_qompiler()

    u = cyclic_matrix(8, 1)
    # execute
    with pytest.raises(EnvironmentError):
        interp.decompose(u)


def test_compile_cyclic_8_ctrl_prune():
    u = cyclic_matrix(8, 1)
    override = dict(emit="CTRL_PRUNED", ancilla_offset=2)
    config = ConfigManager().merge(override).create_config()
    factory = QFactory(config)
    interp = factory.get_qompiler()

    # execute
    bc = interp.decompose(u)
    # print(bc)
    assert bc is not None
    data = [a.data for a in BytecodeIter(bc)]
    assert len(data) == 559
    leaves = [a.data for a in BytecodeIter(bc) if a.is_leaf()]
    v1 = reduce(lambda a, b: a @ b, leaves)
    v = v1.dela().inflate()
    assert np.allclose(v, u), f'circuit != input:\ncompiled=\n{formatter.tostr(v)},\ninput=\n{formatter.tostr(u)}'


def test_compile_cyclic_8():
    n = 3
    u = cyclic_matrix(1 << n, 1)
    override = dict(emit="SINGLET", ancilla_offset=n)
    config = ConfigManager().merge(override).create_config()
    factory = QFactory(config)
    interp = factory.get_qompiler()

    # execute
    bc = interp.decompose(u)
    # print(bc)
    assert bc is not None
    data = [a.data for a in BytecodeIter(bc)]
    assert len(data) == 21
    leaves = [a.data.inflate() for a in BytecodeIter(bc) if a.is_leaf()]
    v = reduce(lambda a, b: a @ b, leaves)
    assert np.allclose(v, u), f'circuit != input:\ncompiled=\n{formatter.tostr(v)},\ninput=\n{formatter.tostr(u)}'


def test_compile_cyclic_4():
    u = cyclic_matrix(4, 1)
    config = ConfigManager().merge(dict(emit="SINGLET")).create_config()
    factory = QFactory(config)
    interp = factory.get_qompiler()

    # execute
    bc = interp.decompose(u)
    # print(bc)
    assert bc is not None
    data = [a.data for a in BytecodeIter(bc)]
    assert len(data) == 7
    leaves = [a.data.inflate() for a in BytecodeIter(bc) if a.is_leaf()]
    assert len(leaves) == 4
    v = reduce(lambda a, b: a @ b, leaves)
    assert np.allclose(v, u), f'circuit != input:\ncompiled=\n{formatter.tostr(v)},\ninput=\n{formatter.tostr(u)}'


def test_interp_random_unitary():
    for _ in range(10):
        # print(f'Test {_}th round')
        n = random.randint(1, 4)
        dim = 1 << n
        u = random_unitary(dim)
        config = ConfigManager().merge(dict(emit="SINGLET", ancilla_offset=n)).create_config()
        factory = QFactory(config)
        interp = factory.get_qompiler()

        # execute
        bc = interp.decompose(u)

        # verify
        leaves = [a.data.inflate() for a in BytecodeIter(bc) if isinstance(a.data, CtrlGate)]
        v = reduce(lambda a, b: a @ b, leaves)
        assert np.allclose(v, u), f'circuit != input:\ncompiled=\n{formatter.tostr(v)},\ninput=\n{formatter.tostr(u)}'


def test_optimize_no_optimizer():
    config = ConfigManager().merge(dict(emit="SINGLET")).create_config()
    factory = QFactory(config)
    compiler = factory.get_qompiler()

    u = random_unitary(2)
    code = compiler.decompose(u)

    # execute
    optcode = compiler.optimize(code)
    assert optcode is not None
    leaves = [a.data for a in BytecodeIter(optcode) if a.is_leaf()]
    v = reduce(lambda a, b: a @ b, leaves)
    assert np.allclose(v.inflate(), u), f'circuit != input:\ncompiled=\n{formatter.tostr(v.inflate())},\ninput=\n{formatter.tostr(u)}'


def test_optimize_basic_optimizer():
    config = ConfigManager().merge(dict(emit="SINGLET")).create_config()
    factory = QFactory(config)
    compiler = factory.get_qompiler()
    u = random_unitary(2)
    code = compiler.decompose(u)

    # execute
    optcode = compiler.optimize(code)
    assert optcode is not None
    leaves = [a.data for a in BytecodeIter(optcode) if a.is_leaf()]
    v = reduce(lambda a, b: a @ b, leaves)
    assert np.allclose(v.inflate(), u), f'circuit != input:\ncompiled=\n{formatter.tostr(v.inflate())},\ninput=\n{formatter.tostr(u)}'


def test_output():
    n = 1
    with tempfile.NamedTemporaryFile(suffix=CODE_FILE_EXT, mode="w+", delete=True) as tmp:
        dim = 1 << n
        u = random_unitary(dim)
        config = ConfigManager().merge(dict(emit="SINGLET", ancilla_offset=n, output=tmp.name)).create_config()
        factory = QFactory(config)
        interp = factory.get_qompiler()

        # execute
        bc = interp.decompose(u)
        interp.output(bc)
        assert os.path.exists(tmp.name)
        actual_size = os.path.getsize(tmp.name)
        assert actual_size == 782
