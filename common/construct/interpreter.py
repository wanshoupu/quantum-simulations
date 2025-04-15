from circuit import CircuitBuilder
from common.construct.bytecode import Bytecode, BytecodeIter
from common.construct.circuit import CirqBuilder
from common.construct.cmat import UnitaryM
from common.construct.quompiler import quompile
from common.utils.mgen import cyclic_matrix


class UnitaryQuantumCircuitInterpreter:

    def __init__(self, builder: CircuitBuilder):
        self.builder = builder

    def interpret(self, component: Bytecode) -> object:
        for c in BytecodeIter(component):
            print(c)


if __name__ == '__main__':
    u = cyclic_matrix(8, 1)
    bc = quompile(u)

    builder = CirqBuilder()
    interpreter = UnitaryQuantumCircuitInterpreter(builder)
    interpreter.interpret(bc)
