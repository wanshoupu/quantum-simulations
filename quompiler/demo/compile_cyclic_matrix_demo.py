from quompiler.circuits.factory_manager import FactoryManager
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import cyclic_matrix

if __name__ == '__main__':
    formatter = MatrixFormatter(precision=2)
    n = 3
    dim = 1 << n
    u = cyclic_matrix(dim, 1)
    print(formatter.tostr(u))
    fman = FactoryManager()
    factory = fman.create_factory()
    interp = factory.get_qompiler()

    # execute
    interp.interpret(u)
    circuit = interp.finish()
    print(circuit)
