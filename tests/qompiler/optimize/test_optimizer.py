import os
from functools import reduce

import numpy as np
from matplotlib import pyplot as plt
from qiskit import transpile
from qiskit.converters import circuit_to_dag

from quompiler.circuits.qfactory import QFactory
from quompiler.config.config_manager import ConfigManager
from quompiler.construct.bytecode import BytecodeIter
from quompiler.construct.types import QompilePlatform
from quompiler.utils.file_io import read_code
from tests.qompiler.circuits.test_qompiler import formatter


def test_optimize():
    codefile = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "for_optimize_test.qco"))
    code = read_code(codefile)
    nodes_before_opts = [a.data for a in BytecodeIter(code) if a.is_leaf()]
    assert len(nodes_before_opts) == 534

    config = ConfigManager().merge(dict(optimization='O3')).create_config()
    factory = QFactory(config)

    # debug
    # print_circuit(codefile, factory)

    # execute
    for opt in factory.get_optimizers():
        code = opt.optimize(code)

    nodes_after_opts = [a.data for a in BytecodeIter(code) if a.is_leaf()]

    # verify
    assert len(nodes_after_opts) == 534

    before_opt = np.array(reduce(lambda a, b: a @ b, nodes_before_opts))
    after_opt = np.array(reduce(lambda a, b: a @ b, nodes_after_opts))
    assert before_opt.shape == after_opt.shape
    assert np.allclose(after_opt, before_opt), f'\ncompiled=\n{formatter.tostr(after_opt)},\ninput=\n{formatter.tostr(before_opt)}'


def print_circuit(codefile, factory):
    render = factory.get_render(QompilePlatform.QISKIT)
    circuit = render.render(codefile)

    # Qiskit built-in optimization
    circuit = transpile(circuit, optimization_level=1)

    dag = circuit_to_dag(circuit)
    layers = list(dag.layers())
    assert len(layers) == 304
    circuit.draw('mpl')
    plt.savefig("qc_qiskit_sketch.pdf", bbox_inches='tight')
