import argparse
import tempfile

from qiskit.converters import circuit_to_dag

from quompiler.circuits.qfactory import QFactory
from quompiler.config.config_manager import ConfigManager, create_config
from quompiler.construct.types import QompilePlatform
from quompiler.utils.file_io import CODE_FILE_EXT
from quompiler.utils.mgen import random_unitary


def compile_random_unitary(filename):
    parser = argparse.ArgumentParser(description="Process some input arguments.")
    parser.add_argument("-i", "--input", type=str, help="Number of qubits. CAUTION: do not set n to big numbers as it scales exponentially.", required=False, default=3)
    args, unknown = parser.parse_known_args()
    n = int(args.input)
    dim = 1 << n
    u = random_unitary(dim)
    override = {"output": filename}
    config_man = ConfigManager().merge(override).parse_args()
    factory = QFactory(config_man.create_config())
    compiler = factory.get_qompiler()
    compiler.compile(u)


def render_cirq(filename):
    config = create_config(emit="CLIFFORD_T", ancilla_offset=100, target="CIRQ", output=filename)
    factory = QFactory(config)
    render = factory.get_render(QompilePlatform.CIRQ)
    codefile = factory.get_config().output
    circuit = render.render(codefile)
    print(circuit)
    moments = circuit.moments
    # for m in moments:
    #     print(m)
    print(f'Total {len(moments)} moments in the circuit.')


def render_qiskit(filename):
    config_man = create_config(emit="CLIFFORD_T", ancilla_offset=100, target="QISKIT", output=filename)
    factory = QFactory(config_man.create_config())
    render = factory.get_render(QompilePlatform.QISKIT)
    codefile = factory.get_config().output
    circuit = render.render(codefile)
    print(circuit)
    dag = circuit_to_dag(circuit)
    layers = list(dag.layers())
    # for m in layers:
    #     print(m)
    print(f'Total {len(layers)} layers in the circuit.')


if __name__ == '__main__':
    with tempfile.NamedTemporaryFile(suffix=CODE_FILE_EXT, mode="w+", delete=True) as tmp:
        compile_random_unitary(tmp.name)
        render_cirq(tmp.name)
        render_qiskit(tmp.name)
