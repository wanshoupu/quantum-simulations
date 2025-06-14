import os
from functools import reduce

import numpy as np
import pytest
from matplotlib import pyplot as plt

from quompiler.circuits.qfactory import QFactory
from quompiler.config.config_manager import ConfigManager
from quompiler.construct.bytecode import BytecodeIter, Bytecode
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.types import QompilePlatform, UnivGate, QType, EmitType
from quompiler.optimize.basic_optimizer import SlidingWindowOptimizer
from quompiler.utils.mgen import random_unitary, random_ctrlgate, create_bytecode
from tests.qompiler.circuits.test_qompiler import formatter


def create_palindromic_code(gates):
    palindrome = gates + [g.herm() for g in gates[::-1]]
    product = reduce(lambda x, y: x @ y, palindrome)
    return Bytecode(product, [Bytecode(g) for g in palindrome])


def test_identity_annihilation_multi_target():
    gates = [CtrlGate(UnivGate.X, [QType.TARGET, QType.CONTROL1]),
             random_ctrlgate(4, 2),
             CtrlGate(UnivGate.X, [QType.TARGET, QType.CONTROL1, QType.CONTROL0])]
    code = create_palindromic_code(gates)
    assert np.allclose(np.array(code.data), np.eye(code.data.order()))

    gates_before_opts = [a.data for a in BytecodeIter(code) if a.is_leaf()]

    length_before_opts = len(gates_before_opts)

    # execute
    opt = SlidingWindowOptimizer(length_before_opts, emit=EmitType.CLIFFORD_T)
    code = opt.optimize(code)
    nodes_after_opts = [a for a in BytecodeIter(code) if a.is_leaf() and not a.skip]

    # verify
    assert len(nodes_after_opts) == 0


@pytest.mark.parametrize('seq, ctrl_num, emit', [
    ['X,X', 3, EmitType.SINGLET],
    ['SD,S', 3, EmitType.SINGLET],
    ['T,SD,S,TD', 2, EmitType.CLIFFORD_T],
    ["S,T,TD,SD", 1, EmitType.CLIFFORD_T],
])
def test_identity_annihilation_std(seq, ctrl_num, emit):
    code = create_bytecode(seq, ctrl_num)
    gates_before_opts = [a.data for a in BytecodeIter(code) if a.is_leaf()]
    length_before_opts = len(gates_before_opts)

    # execute
    opt = SlidingWindowOptimizer(length_before_opts, emit=emit)
    code = opt.optimize(code)

    # verify
    nodes_after_opts = [a for a in BytecodeIter(code) if a.is_leaf() and not a.skip]
    # print(nodes_after_opts)
    assert len(nodes_after_opts) == 0


@pytest.mark.parametrize('seq, ctrl_num, emit', [
    ['T,SD,S', 3, EmitType.SINGLET],
    ['SD,S,X', 2, EmitType.CLIFFORD_T],
])
def test_identity_reduce_std(seq, ctrl_num, emit):
    code = create_bytecode(seq, ctrl_num)
    gates_before_opts = [a.data for a in BytecodeIter(code) if a.is_leaf()]
    length_before_opts = len(gates_before_opts)

    # execute
    opt = SlidingWindowOptimizer(length_before_opts, emit=emit)
    code = opt.optimize(code)

    # verify
    nodes_after_opts = [a for a in BytecodeIter(code) if a.is_leaf() and not a.skip]
    # print(nodes_after_opts)
    assert len(nodes_after_opts) == 1
    expected = np.array(reduce(lambda a, b: a @ b, gates_before_opts))
    actual = np.array(nodes_after_opts[0].data)
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected), f'\noptimized=\n{formatter.tostr(actual)},\nexpected=\n{formatter.tostr(expected)}'


@pytest.mark.parametrize('seq, ctrl_num, emit, count', [
    ['T,SD', 3, EmitType.SINGLET, 1],
    ['T,SD,S,T', 2, EmitType.CLIFFORD_T, 1],
    ["S,T,SD", 1, EmitType.CLIFFORD_T, 1],
    ["S,T,SD,TD", 1, EmitType.CLIFFORD_T, 0],
])
def test_merge_std(seq, ctrl_num, emit, count):
    code = create_bytecode(seq, ctrl_num)
    gates_before_opts = [a.data for a in BytecodeIter(code) if a.is_leaf()]
    length_before_opts = len(gates_before_opts)

    # execute
    opt = SlidingWindowOptimizer(length_before_opts, emit=emit)
    code = opt.optimize(code)

    # verify
    nodes_after_opts = [a for a in BytecodeIter(code) if a.is_leaf() and not a.skip]
    # print(nodes_after_opts)
    assert len(nodes_after_opts) == count
    expected = np.array(reduce(lambda a, b: a @ b, gates_before_opts))
    gates = [n.data for n in nodes_after_opts]
    actual = np.array(reduce(lambda a, b: a @ b, gates, np.eye(expected.shape[0])))
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected), f'\noptimized=\n{formatter.tostr(actual)},\nexpected=\n{formatter.tostr(expected)}'


def test_optimize_combine_four():
    code = create_bytecode("S,T,SD,TD", 1)

    gates_before_opts = [a.data for a in BytecodeIter(code) if a.is_leaf()]
    # debug
    # print_circuit(codefile, factory)

    assert len(gates_before_opts) == 4
    config = ConfigManager().merge(dict(emit='PRINCIPAL', ancilla_offset=8, optimization='O3')).create_config()
    factory = QFactory(config)

    # execute
    for opt in factory.get_optimizers():
        code = opt.optimize(code)

    nodes_after_opts = [a for a in BytecodeIter(code) if a.is_leaf() and not a.skip]

    # verify
    assert len(nodes_after_opts) == 0


def test_optimize_real_compile():
    # with tempfile.NamedTemporaryFile(suffix=CODE_FILE_EXT, mode="w+b", delete=True) as codefile:
    codefile = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "for_optimize_test.qco"))
    n = 3

    config = ConfigManager().merge(dict(emit='PRINCIPAL', ancilla_offset=n, optimization='O3', output=codefile)).create_config()
    factory = QFactory(config)

    dim = 1 << n
    input_mat = random_unitary(dim)
    compiler = factory.get_qompiler()
    root = compiler.decompose(input_mat)
    code = root.children[5]
    compiler.output(code)

    gates_before_opts = [a.data for a in BytecodeIter(code) if a.is_leaf()]
    # debug
    # print_circuit(codefile, factory, "before_opt.pdf")

    assert len(gates_before_opts) == 126
    # execute
    for opt in factory.get_optimizers():
        code = opt.optimize(code)

    nodes_after_opts = [a for a in BytecodeIter(code) if a.is_leaf() and not a.skip]

    # verify
    assert len(nodes_after_opts) < len(gates_before_opts)

    compiler.output(code)
    # debug
    # print_circuit(codefile, factory, "after_opt.pdf")

    gates_after_opts = [a.data for a in nodes_after_opts]
    expected = np.array(reduce(lambda a, b: a @ b, gates_before_opts))
    actual = np.array(reduce(lambda a, b: a @ b, gates_after_opts))
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected), f'\noptimized=\n{formatter.tostr(actual)},\nexpected=\n{formatter.tostr(expected)}'


def print_circuit(codefile, factory, filepath):
    render = factory.get_render(QompilePlatform.QISKIT)
    circuit = render.render(codefile)

    # Qiskit built-in optimization
    circuit.draw('mpl')
    plt.savefig(filepath, bbox_inches='tight')
    # plt.show()
