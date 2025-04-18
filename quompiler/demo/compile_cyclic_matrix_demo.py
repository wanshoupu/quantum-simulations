from quompiler.circuits.cirq_circuit import CirqBuilder
from quompiler.circuits.interpreter import CircuitInterp
from quompiler.construct.quompiler import quompile
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import cyclic_matrix

if __name__ == '__main__':
    formatter = MatrixFormatter(precision=2)
    n = 3
    dim = 1 << n
    u = cyclic_matrix(dim, 1)
    print(formatter.tostr(u))
    builder = CirqBuilder(n)
    CircuitInterp(builder).interpret(u)
    circuit = builder.finish()
    print(circuit)
