import argparse
import tempfile

from qiskit.converters import circuit_to_dag

from quompiler.circuits.factory_manager import FactoryManager
from quompiler.circuits.render import QRenderer
from quompiler.utils.file_io import CODE_FILE_EXT
from quompiler.utils.mgen import random_unitary
from tests.qompiler.mock_fixtures import mock_factory_manager, mock_config


def compile_random_unitary(filename):
    parser = argparse.ArgumentParser(description="Process some input arguments.")
    parser.add_argument("-i", "--input", type=str, help="Number of qubits. CAUTION: do not set n to big numbers as it scales exponentially.", required=False, default=3)
    args, unknown = parser.parse_known_args()
    n = int(args.input)
    dim = 1 << n
    u = random_unitary(dim)
    man = FactoryManager()
    override = {"output": filename}
    man.merge(override)
    factory = man.create_factory()
    compiler = factory.get_qompiler()
    compiler.compile(u)


def render_cirq(filename):
    man = mock_factory_manager(emit="CLIFFORD_T", ancilla_offset=100, target="CIRQ")
    override = {"output": filename}
    man.merge(override)
    factory = man.create_factory()
    render = QRenderer(man.create_config(), factory.get_builder())
    codefile = factory.get_config().output
    circuit = render.render(codefile)
    print(circuit)
    moments = circuit.moments
    # for m in moments:
    #     print(m)
    print(f'Total {len(moments)} moments in the circuit.')


def render_qiskit(filename):
    man = mock_factory_manager(emit="CLIFFORD_T", ancilla_offset=100, target="QISKIT")
    override = {"output": filename}
    man.merge(override)
    factory = man.create_factory()
    render = QRenderer(man.create_config(), factory.get_builder())
    codefile = factory.get_config().output
    circuit = render.render(codefile)
    print(circuit)
    dag = circuit_to_dag(circuit)
    layers = list(dag.layers())
    # for m in moments:
    #     print(m)
    print(f'Total {len(layers)} layers in the circuit.')


if __name__ == '__main__':
    with tempfile.NamedTemporaryFile(suffix=CODE_FILE_EXT, mode="w+", delete=True) as tmp:
        compile_random_unitary(tmp.name)
        render_cirq(tmp.name)
        render_qiskit(tmp.name)
