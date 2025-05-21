"""
This module provide the compilation functionalities.
If needed, it may make distinctions between target qubits and ancilla qubits.
"""

from typing import Union

from numpy._typing import NDArray

from quompiler.circuits.qbuilder import CircuitBuilder
from quompiler.circuits.qdevice import QDevice
from quompiler.config.construct import QompilerConfig
from quompiler.construct.bytecode import BytecodeRevIter, Bytecode
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.types import EmitType, UnivGate
from quompiler.construct.unitary import UnitaryM
from quompiler.utils.cnot_decompose import cnot_decompose
from quompiler.utils.granularity import granularity
from quompiler.utils.mat2l_decompose import mat2l_decompose
from quompiler.utils.std_decompose import std_decompose, ctrl_decompose


class Qompiler:

    def __init__(self, config: QompilerConfig, builder: CircuitBuilder, device: QDevice):
        self.config = config
        self.builder = builder
        self.device = device
        self.emit = EmitType[config.emit]

    def interpret(self, u: NDArray):
        component = self.compile(u)
        for c in BytecodeRevIter(component):
            m = c.data
            if isinstance(m, CtrlGate):
                self.builder.build_gate(m)
            elif isinstance(m, UnitaryM):
                self.builder.build_group(m)

    def compile(self, u: NDArray) -> Bytecode:
        s = u.shape
        um = UnitaryM(s[0], tuple(range(s[0])), u)
        return self._decompose(um)

    def _decompose(self, data: Union[UnitaryM, CtrlGate]) -> Bytecode:
        root = Bytecode(data)
        grain = granularity(data)
        if self.emit <= grain:  # noop
            return root

        if isinstance(data, UnitaryM):
            constituents = self._decompose_unitary(grain, data)
        elif isinstance(data, CtrlGate):
            constituents = self._decompose_ctrl(grain, data)
        else:
            raise ValueError(f"Unrecognized gate of type {type(grain)}")
        # decompose is noop
        if len(constituents) == 1 and constituents[0] == data:
            return root
        for c in constituents:
            root.append(self._decompose(c))
        return root

    def _decompose_std(self, gate: CtrlGate) -> list[CtrlGate]:
        std_gates = UnivGate.cliffordt() if self.emit == EmitType.CLIFFORD_T else list(UnivGate)
        constituents = std_decompose(gate, std_gates, self.config.rtol, self.config.atol)
        return constituents

    def _decompose_ctrl(self, grain: EmitType, gate: CtrlGate) -> list[CtrlGate]:
        # EmitType.MULTI_TARGET is disabled atm
        # if g < EmitType.MULTI_TARGET:
        #     result = ctrl_decompose(u, clength=2, aspace=self.aspace)
        if grain < EmitType.CTRL_PRUNED:
            result = ctrl_decompose(gate, self.device, clength=1)
        else:
            result = self._decompose_std(gate)
        return result

    @staticmethod
    def _decompose_unitary(grain: EmitType, u: UnitaryM) -> list[Union[UnitaryM, CtrlGate]]:
        if grain < EmitType.TWO_LEVEL:
            result = mat2l_decompose(u)
        else:
            assert u.is2l()
            result = cnot_decompose(u)
        return result

    def finish(self, optimized=False) -> object:
        return self.builder.finish(optimized=optimized)

    def all_qubits(self):
        return self.builder.all_qubits()
