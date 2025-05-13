from abc import ABC, abstractmethod

from quompiler.config.construct import DeviceConfig
from quompiler.construct.qspace import Ancilla, Qubit


class QDevice(ABC):
    """
    Represent an abstract Quantum Device.
    """

    def __init__(self, config: DeviceConfig):
        self.CAPACITY = 2 * config.ancilla_offset  # heuristic upper limit of the qspace
        self.qoffset = 0
        self.aoffset = config.ancilla_offset
        self.aspace = []
        self.qspace = []

    def alloc_qubit(self, n):
        if self.qoffset + n > self.CAPACITY:
            raise EnvironmentError(f'Not enough qubits to allocate ancilla {n} qubits')
        self.qspace.extend(Qubit(self.qoffset + i) for i in range(n))
        self.qoffset += n
        return self.qspace[-n:]

    def alloc_ancilla(self, n):
        if self.qoffset + n > self.CAPACITY:
            raise EnvironmentError(f'Not enough qubits to allocate ancilla {n} qubits')
        self.aspace.extend(Ancilla(i) for i in range(self.aoffset, n))
        self.aoffset += n
        while len(self.aspace) < n:
            self.aoffset += 1
            self.aspace.append(Ancilla(self.aoffset))
        result = self.aspace[:n]
        for a in result:
            self.reset(a)
        return result

    @abstractmethod
    def reset(self, qubit: Qubit):
        pass

    @abstractmethod
    def map(self, qid: Qubit):
        """
        Create the physical qubits for the given qid.
        :param qid:
        :return:
        """
        pass
