from abc import ABC, abstractmethod

from quompiler.config.construct import DeviceConfig
from quompiler.construct.qspace import Ancilla, Qubit


class QDevice(ABC):
    """
    Represent an abstract Quantum Device.
    """

    def __init__(self, config: DeviceConfig):
        self.CAPACITY = 2 * config.ancilla_offset  # heuristic upper limit of the qspace
        self.offset = config.ancilla_offset
        self.qoffset = 0
        self.aoffset = self.offset
        self.aspace = []
        self.qspace = []

    def alloc_ancilla(self, n):
        while len(self.aspace) < n:
            self.aoffset += 1
            self.qspace.append(Ancilla(self.aoffset))
        result = self.aspace[:n]
        for a in result:
            self.reset(a)
        return result

    def reset(self, qubit):
        pass

    @abstractmethod
    def map(self, qid: Qubit):
        """
        Create the physical qubits for the given qid.
        :param qid:
        :return:
        """
        pass
