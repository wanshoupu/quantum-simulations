import random

from common.circuits.cirq_circuit import CirqBuilder
from common.circuits.interpreter import CircuitInterp
from common.construct.quompiler import quompile
from common.utils.format_matrix import MatrixFormatter
from common.utils.mgen import random_unitary
import numpy as np

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)

    n = random.randint(1, 4)
    dim = 1 << n
    u = random_unitary(dim)
    bc = quompile(u)

    builder = CirqBuilder(n)
    interpreter = CircuitInterp(builder)
    interpreter.interpret(bc)
    circuit = builder.finish()
    print(circuit)
    moments = circuit.moments
    for m in moments:
        print(m)
    print(f'Total {len(moments)} moments in the circuit.')
