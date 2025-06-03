import argparse

from quompiler.circuits.factory_manager import FactoryManager
from quompiler.utils.mgen import random_unitary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some input arguments.")
    parser.add_argument("-i", "--input", type=str, help="Number of qubits. CAUTION: do not set n to big numbers as it scales exponentially.", required=False, default=3)
    args, unknown = parser.parse_known_args()

    n = int(args.input)
    dim = 1 << n
    u = random_unitary(dim)

    man = FactoryManager()
    man.parse_args()

    factory = man.create_factory()
    compiler = factory.get_qompiler()
    compiler.compile(u)

    render = factory.get_render()
    codefile = factory.get_config().output
    circuit = render.render(codefile)
    print(circuit)
    moments = circuit.moments
    # for m in moments:
    #     print(m)
    print(f'Total {len(moments)} moments in the circuit.')
