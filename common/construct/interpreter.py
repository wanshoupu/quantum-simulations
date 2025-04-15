from typing import List

import numpy as np
from numpy.typing import NDArray
from sympy.physics.quantum import represent
from sympy.physics.quantum.gate import CGate, XGate
from typing_extensions import override

from circuit import Circuit, CircuitBuilder
from common.construct.bytecode import Bytecode
from common.construct.cmat import CUnitary, UnitaryM


class UnitaryInterpreter:

    def __init__(self, builder: CircuitBuilder):
        self.builder = builder

    def interpret(self, component: Bytecode) -> object:
        qc = Circuit()


if __name__ == '__main__':
    x = XGate(1)
    m = represent(x, nqubits=2)
    print(m)
    print(np.array(m))
