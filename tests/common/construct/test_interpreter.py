from common.construct.circuit import CirqBuilder
from common.construct.interpreter import CircuitInterp
from common.construct.quompiler import quompile
from common.utils.mgen import cyclic_matrix


def test_interp():
    u = cyclic_matrix(8, 1)
    bc = quompile(u)

    builder = CirqBuilder()
    interpreter = CircuitInterp(builder)
    interpreter.interpret(bc)
