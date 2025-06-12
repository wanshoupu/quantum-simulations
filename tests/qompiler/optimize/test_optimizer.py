import os
import tempfile
from functools import reduce

import numpy as np
from matplotlib import pyplot as plt

from quompiler.circuits.qfactory import QFactory
from quompiler.config.config_manager import ConfigManager
from quompiler.construct.bytecode import BytecodeIter, Bytecode
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.types import QompilePlatform, UnivGate, QType
from quompiler.utils.file_io import CODE_FILE_EXT
from quompiler.utils.mgen import random_unitary, random_ctrlgate
from tests.qompiler.circuits.test_qompiler import formatter


def create_palindromic_code():
    gates = [CtrlGate(UnivGate.X, [QType.TARGET, QType.CONTROL1]),
             (random_ctrlgate(4, 2)),
             CtrlGate(UnivGate.X, [QType.TARGET, QType.CONTROL1, QType.CONTROL0]),
             ]
    gates = gates + [gates[-1], gates[-2].herm(), gates[-3]]
    product = reduce(lambda x, y: x @ y, gates)
    return Bytecode(product, [Bytecode(g) for g in gates])


def test_optimize_palindrome():
    code = create_palindromic_code()
    assert np.allclose(np.array(code.data), np.eye(code.data.order()))

    gates_before_opts = [a.data for a in BytecodeIter(code) if a.is_leaf()]
    # debug
    # print_circuit(codefile, factory)

    assert len(gates_before_opts) == 6
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
    print_circuit(codefile, factory, "before_opt.pdf")

    assert len(gates_before_opts) == 126
    # execute
    for opt in factory.get_optimizers():
        code = opt.optimize(code)

    nodes_after_opts = [a for a in BytecodeIter(code) if a.is_leaf() and not a.skip]

    # verify
    assert len(nodes_after_opts) == 108

    compiler.output(code)
    print_circuit(codefile, factory, "after_opt.pdf")

    gates_after_opts = [a.data for a in nodes_after_opts]
    expected = np.array(reduce(lambda a, b: a @ b, gates_before_opts))
    actual = np.array(reduce(lambda a, b: a @ b, gates_after_opts))
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected), f'\noptimized=\n{formatter.tostr(actual)},\nexpected=\n{formatter.tostr(expected)}'


def print_circuit(codefile, factory, filepath="qc_qiskit_sketch.pdf"):
    render = factory.get_render(QompilePlatform.QISKIT)
    circuit = render.render(codefile)

    # Qiskit built-in optimization
    circuit.draw('mpl')
    plt.savefig(filepath, bbox_inches='tight')
    # plt.show()
