import quimb.tensor as qtn
from typing_extensions import override

from common.construct.circuit import CircuitBuilder
from common.construct.cmat import UnitaryM, validm2l


class QuimbBuilder(CircuitBuilder):
    @override
    def __init__(self, dimension: int):
        self.qubits = []
        self.circuit = qtn.circuit.Circuit(dimension)
        self.counter = 1

    @override
    def get_unigate(self, m: UnitaryM) -> object:
        pass

    @override
    def build_gate(self, m: UnitaryM):
        self.counter += 1
        if not validm2l(m.matrix):
            custom_gate = qtn.circuit.Gate(str(self.counter), m.matrix)
            # self.circuit.apply_gate(self.counter,'H', 0)
            self.circuit.apply_gate(custom_gate, self.qubits)

        # if isinstance(m, CUnitary):

    @override
    def finish(self) -> qtn.circuit.Circuit:
        pass
