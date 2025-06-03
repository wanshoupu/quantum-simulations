from quompiler.config.construct import DeviceConfig
from quompiler.construct.qspace import Qubit


class QDevice:
    """
    Represent an virtual Quantum Device.
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
        self.aspace.extend(Qubit(i, ancilla=True) for i in range(self.aoffset, n))
        self.aoffset += n
        while len(self.aspace) < n:
            self.aoffset += 1
            self.aspace.append(Qubit(self.aoffset, ancilla=True))
        return self.aspace[:n]
