import argparse
import os

from quompiler.circuits.create_factory import create_factory
from quompiler.config.construct import QompilerConfig
from quompiler.circuits.qompiler import Qompiler
from quompiler.utils.mgen import random_unitary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some input arguments.")
    parser.add_argument("-i", "--input", type=str, help="Number of qubits. CAUTION: do not set n to big numbers as it scales exponentially.", required=False, default=3)
    args = parser.parse_args()

    n = int(args.input)
    dim = 1 << n
    u = random_unitary(dim)

    cfile = os.path.abspath(os.path.join(os.path.dirname(__file__), "compiler_config.json"))
    config = QompilerConfig.from_file(cfile)
    interp = Qompiler(config)
    interp.interpret(u)
    circuit = interp.finish(optimized=True)
    print(circuit)
    moments = circuit.moments
    # for m in moments:
    #     print(m)
    print(f'Total {len(moments)} moments in the circuit.')
