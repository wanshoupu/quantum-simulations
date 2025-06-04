from typing import Union, Optional, Sequence

from quimb import tensor as qtn
from typing_extensions import override

from quompiler.circuits.qbuilder import CircuitBuilder
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.qspace import Qubit
from quompiler.construct.types import UnivGate
from quompiler.construct.unitary import UnitaryM
from quompiler.utils.mat_utils import validm2l


class QuimbBuilder(CircuitBuilder):

    def __init__(self):
        self.qubits = []
        self.circuit = qtn.circuit.Circuit(8)
        self.counter = 1

    def get_univ_gate(self, m: Union[UnitaryM, CtrlGate]) -> Optional[UnivGate]:
        pass

    def build_gate(self, m: Union[UnitaryM, CtrlGate]):
        self.counter += 1
        if not validm2l(m.matrix):
            custom_gate = qtn.circuit.Gate(str(self.counter), m.matrix)
            # self.circuit.apply_gate(self.counter,'H', 0)
            self.circuit.apply_gate(custom_gate, self.qubits)

    @override
    def register(self, qspace: Sequence[Qubit]) -> None:
        pass

    @override
    def finish(self, optimized=False) -> qtn.circuit.Circuit:
        pass

    @override
    def all_qubits(self) -> list:
        pass
