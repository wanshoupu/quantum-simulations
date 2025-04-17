import random

import numpy as np
from cirq import Circuit, merge_single_qubit_gates_to_phased_x_and_z, eject_z, drop_negligible_operations, drop_empty_moments

from quompiler.circuits.cirq_circuit import CirqBuilder
from quompiler.circuits.interpreter import CircuitInterp
from quompiler.construct.quompiler import quompile
from quompiler.utils.mgen import random_unitary


def optimize(circuit: Circuit):
    circuit = merge_single_qubit_gates_to_phased_x_and_z(circuit)
    circuit = eject_z(circuit)
    circuit = drop_negligible_operations(circuit)
    circuit = drop_empty_moments(circuit)
    return circuit


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)

    # CAUTION: do not set n to big numbers as it scales exponentially.
    n = 4
    dim = 1 << n
    u = random_unitary(dim)
    bc = quompile(u)

    builder = CirqBuilder(n)
    interpreter = CircuitInterp(builder)
    interpreter.interpret(bc)
    circuit = builder.finish()
    circuit = optimize(circuit)
    print(circuit)
    moments = circuit.moments
    # for m in moments:
    #     print(m)
    print(f'Total {len(moments)} moments in the circuit.')
