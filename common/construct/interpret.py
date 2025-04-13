from typing import List
from abc import ABC, abstractmethod
import numpy as np
from sympy.physics.quantum.gate import Gate, CGate
from typing_extensions import override
from numpy.typing import NDArray

from circuit import Circuit


class UnitaryInterpreter(ABC):
    def gate(self, m: NDArray):
        """
        Interpret the matrix m as a controlled gate targeting a single qubit
        :param m:
        :return:
        """
        t = self.target(m)
        meta = CGate

    def target(self, m: NDArray) -> int:

        pass

    @abstractmethod
    def circuit(self, n: int):
        pass

    def interpret(self, component: List[NDArray]) -> object:
        qc = Circuit()


class CirqUnitaryInterpreter(UnitaryInterpreter):
    @override
    def gate(self, m: NDArray):
        meta = CGate

    @override
    def circuit(self, n: int):
        pass

    def configure(self):
        pass


class QiskitUnitaryInterpreter(UnitaryInterpreter):

    def gate(self, m: NDArray):
        pass

    def circuit(self, n: int):
        pass

    def configure(self):
        pass


if __name__ == '__main__':
    pass
