from abc import ABC, abstractmethod
from typing import List

from numpy._typing import NDArray

from common.construct.cmat import CUnitary, UnitaryM
from sympy.physics.quantum.gate import CGate, XGate

from typing_extensions import override


class Circuit:
    pass


class CircuitBuilder(ABC):
    @abstractmethod
    def gate(self, m: UnitaryM):
        """
        Interpret the matrix m as a controlled gate targeting a single qubit
        :param m:
        :return:
        """
        if isinstance(m, CUnitary):
            meta = CGate()

    @abstractmethod
    def circuit(self):
        pass

    @abstractmethod
    def configure(self):
        pass


class CirqUnitaryInterpreter(CircuitBuilder):
    @override
    def gate(self, m: UnitaryM):
        meta = CGate

    @override
    def circuit(self, n: int):
        pass

    @override
    def configure(self):
        pass


class QiskitUnitaryInterpreter(CircuitBuilder):

    @override
    def gate(self, m: NDArray):
        pass

    @override
    def circuit(self, n: int):
        pass

    @override
    def configure(self):
        pass
