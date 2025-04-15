from abc import ABC, abstractmethod

import cirq
from sympy.physics.quantum.gate import CGate
from typing_extensions import override

from common.construct.cmat import CUnitary, UnitaryM


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
    def finish(self):
        pass


class CirqBuilder(CircuitBuilder):

    def __init__(self):
        self.qubits = []
        self.circuit = cirq.Circuit()

    @override
    def gate(self, m: UnitaryM):
        meta = CGate

    @override
    def finish(self):
        pass


class QiskitBuilder(CircuitBuilder):

    @override
    def gate(self, m: UnitaryM):
        pass

    @override
    def finish(self):
        pass
