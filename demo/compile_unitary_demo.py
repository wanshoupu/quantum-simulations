import argparse
import random

import numpy as np
import qiskit.qasm3 as qasm
from matplotlib import pyplot as plt
from qiskit.converters import circuit_to_dag

from quompiler.circuits.cirq_factory.presentation import cirq2svg
from quompiler.circuits.qfactory import QFactory
from quompiler.config.config_manager import create_config
from quompiler.construct.types import QompilePlatform
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_unitary


def compile_random_unitary(filename):
    parser = argparse.ArgumentParser(description="Process some input arguments.")
    parser.add_argument("-i", "--input", type=str, help="Number of qubits. CAUTION: do not set n to big numbers as it scales exponentially.", required=False, default=3)
    args, unknown = parser.parse_known_args()
    n = int(args.input)
    dim = 1 << n
    u = random_unitary(dim)
    print(formatter.tostr(u))
    config = create_config(emit="PRINCIPAL", ancilla_offset=100, output=filename)
    factory = QFactory(config)
    compiler = factory.get_qompiler()
    compiler.compile(u)


def render_cirq(filename):
    config = create_config(target="CIRQ", output=filename)
    factory = QFactory(config)
    render = factory.get_render(QompilePlatform.CIRQ)
    codefile = factory.get_config().output
    circuit = render.render(codefile)
    from cirq import merge_single_qubit_gates_to_phased_x_and_z
    circuit = merge_single_qubit_gates_to_phased_x_and_z(circuit)
    print(circuit.to_text_diagram(transpose=True))
    svg = cirq2svg(circuit)
    with open("cirq_circuit.svg", "w") as f:
        f.write(svg)

    moments = circuit.moments
    # for m in moments:
    #     print(m)
    print(f'Total {len(moments)} moments in the circuit.')


def render_qiskit(filename):
    config = create_config(target="QISKIT", output=filename)
    factory = QFactory(config)
    render = factory.get_render(QompilePlatform.QISKIT)
    codefile = factory.get_config().output
    circuit = render.render(codefile)

    # Export to QASM string
    qasm_code = qasm.dumps(circuit)
    print(qasm_code)

    dag = circuit_to_dag(circuit)
    layers = list(dag.layers())
    # for m in layers:
    #     print(m)
    print(f'Total {len(layers)} layers in the circuit.')
    circuit.draw('mpl')
    qiskit_diagram_file = "qc_qiskit_sketch.pdf"
    plt.savefig(qiskit_diagram_file, bbox_inches='tight')
    print(qiskit_diagram_file)


if __name__ == '__main__':
    formatter = MatrixFormatter(precision=4)
    random.seed(42)
    np.random.seed(42)

    tmp = 'tmp-circuit.qco'
    render_cirq(tmp)

    # with tempfile.NamedTemporaryFile(suffix=CODE_FILE_EXT, mode="w+", delete=True) as tmp:
    #     compile_random_unitary(tmp.name)
    #     render_cirq(tmp.name)
    #     render_qiskit(tmp.name)
