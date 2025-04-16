from common.construct.bytecode import Bytecode, BytecodeIter
from common.circuits.circuit_builder import CircuitBuilder
from common.construct.cmat import CUnitary, UnitaryM


class CircuitInterp:

    def __init__(self, builder: CircuitBuilder):
        self.builder = builder

    def interpret(self, component: Bytecode):
        for c in BytecodeIter(component):
            m = c.data
            if isinstance(m, CUnitary):
                # TODO for now draw single-qubit + controlled single-qubit as gate.
                # TO BE breakdown further to elementary gates only
                self.builder.build_gate(m)
            elif isinstance(m, UnitaryM):
                self.builder.build_group(m)

