from typing import Union, Optional

from quimb import tensor as qtn
from typing_extensions import override

from quompiler.circuits.qbuilder import CircuitBuilder
from quompiler.config.construct import DeviceConfig
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.std_gate import CtrlStdGate
from quompiler.construct.types import UnivGate
from quompiler.construct.unitary import UnitaryM
from quompiler.utils.mat_utils import validm2l


class QuimbBuilder(CircuitBuilder):
    def __init__(self, deviceConfig: DeviceConfig):
        self.qubits = []
        self.circuit = qtn.circuit.Circuit(deviceConfig.dimension)
        self.counter = 1

    def get_univ_gate(self, m: Union[UnitaryM, CtrlGate, CtrlStdGate]) -> Optional[UnivGate]:
        pass

    def build_gate(self, m: Union[UnitaryM, CtrlGate, CtrlStdGate]):
        self.counter += 1
        if not validm2l(m.matrix):
            custom_gate = qtn.circuit.Gate(str(self.counter), m.matrix)
            # self.circuit.apply_gate(self.counter,'H', 0)
            self.circuit.apply_gate(custom_gate, self.qubits)

    @override
    def finish(self) -> qtn.circuit.Circuit:
        pass

    @override
    def all_qubits(self) -> list:
        pass
