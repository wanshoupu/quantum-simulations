import argparse

from quompiler.circuits.cirq_circuit import CirqBuilder
from quompiler.qompile.quompiler import CircuitInterp
from quompiler.construct.quompiler import quompile
from quompiler.utils.mgen import random_unitary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some input arguments.")
    parser.add_argument("-i", "--input", type=str, help="Number of qubits. CAUTION: do not set n to big numbers as it scales exponentially.", required=False, default=3)
    args = parser.parse_args()

    n = int(args.input)
    dim = 1 << n
    u = random_unitary(dim)
    bc = quompile(u)

    builder = CirqBuilder(n)
    CircuitInterp(builder).interpret(u)
    circuit = builder.finish(optimized=True)
    print(circuit)
    moments = circuit.moments
    # for m in moments:
    #     print(m)
    print(f'Total {len(moments)} moments in the circuit.')
