from numpy.typing import NDArray

from quompiler.circuits.circuit_builder import CircuitBuilder
from quompiler.construct.bytecode import ReverseBytecodeIter
from quompiler.construct.cmat import ControlledM
from quompiler.construct.unitary import UnitaryM
from quompiler.construct.quompiler import quompile


class CircuitInterp:

    def __init__(self, builder: CircuitBuilder):
        self.builder = builder

    def interpret(self, u: NDArray):
        component = quompile(u)
        for c in ReverseBytecodeIter(component):
            m = c.data
            if isinstance(m, ControlledM):
                # TODO for now draw single-qubit + controlled single-qubit as gate.
                # TO BE breakdown further to elementary gates only
                self.builder.build_gate(m)
            elif isinstance(m, UnitaryM):
                self.builder.build_group(m)
