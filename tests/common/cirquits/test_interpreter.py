from common.circuits.cirq_circuit import CirqBuilder
from common.circuits.interpreter import CircuitInterp
from common.construct.quompiler import quompile
from common.utils.mgen import cyclic_matrix

import numpy as np


def test_interp_identity_matrix():
    n = 3
    dim = 1 << n
    u = np.eye(dim)
    bc = quompile(u)

    builder = CirqBuilder(n)
    interpreter = CircuitInterp(builder)
    interpreter.interpret(bc)
    print(builder.finish())


def test_interp_cyclic_matrix():
    n = 3
    dim = 1 << n
    u = cyclic_matrix(dim, 1)
    bc = quompile(u)

    builder = CirqBuilder(n)
    interpreter = CircuitInterp(builder)
    interpreter.interpret(bc)
    print(builder.finish())
