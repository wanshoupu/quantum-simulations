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
from quompiler.optimize.optimizer import Optimizer
from quompiler.utils.cnot_decompose import cnot_decompose
from quompiler.utils.granularity import granularity
from quompiler.utils.mat2l_decompose import mat2l_decompose
from quompiler.utils.std_decompose import std_decompose
from quompiler.utils.ctrl_decompose import ctrl_decompose


class Qompiler:

    def __init__(self, config: QompilerConfig, builder: CircuitBuilder, device: QDevice, optimizers: list[Optimizer] = None):
        self.config = config
        self.builder = builder
        self.device = device
        self.optimizers = optimizers or []
        self.emit = EmitType[config.emit]

    def interpret(self, u: NDArray):
        code = self.compile(u)
        code = self.optimize(code)
        self.render(code)

    def optimize(self, code):
        for opt in self.optimizers:
            code = opt.optimize(code)
        return code

    def render(self, code: Bytecode):
        for c in BytecodeRevIter(code):
            m = c.data
            if not c.is_leaf():
                self.builder.build_group(m)
            else:
                self.builder.build_gate(m)

    def compile(self, u: NDArray) -> Bytecode:
        s = u.shape
        um = UnitaryM(s[0], tuple(range(s[0])), u)
        return self._decompose(um)

    def _decompose(self, data: Union[UnitaryM, CtrlGate]) -> Bytecode:
        root = Bytecode(data)
        if isinstance(data, UnitaryM):
            root.metadata['data'] = f'UnitaryM(core={data.core})'
        elif isinstance(data, CtrlGate):
            root.metadata['data'] = f'CtrlGate(ctrl={data.controls},qspace={data.qspace})'

        grain = granularity(data)
        if self.emit <= grain:  # noop
            return root

        if isinstance(data, UnitaryM):
            constituents, meta = self._decompose_unitary(grain, data)
        elif isinstance(data, CtrlGate):
            constituents, meta = self._decompose_ctrl(grain, data)
        else:
            raise ValueError(f"Unrecognized gate of type {type(grain)}")
        root.metadata.update(meta)
        root.metadata['fanout'] = len(constituents)

        # decompose is noop
        if len(constituents) == 1 and constituents[0] == data:
            return root
        for c in constituents:
            root.append(self._decompose(c))
        return root

    def _decompose_std(self, gate: CtrlGate) -> tuple[list[CtrlGate], dict]:
        constituents = std_decompose(gate, self.emit, self.config.rtol, self.config.atol)
        return constituents, {'method': 'std_decompose', 'params': str(self.emit)}

    def _decompose_ctrl(self, grain: EmitType, gate: CtrlGate) -> tuple[list[CtrlGate], dict]:
        meta = dict()
        # EmitType.MULTI_TARGET is disabled atm
        # if g < EmitType.MULTI_TARGET:
        #     result = ctrl_decompose(u, clength=2, aspace=self.aspace)
        if grain < EmitType.CTRL_PRUNED:
            result = ctrl_decompose(gate, self.device, clength=1)
            meta['method'] = 'ctrl_decompose'
        else:
            result, meta1 = self._decompose_std(gate)
            meta.update(meta1)
        return result, meta

    @staticmethod
    def _decompose_unitary(grain: EmitType, u: UnitaryM) -> tuple[list[Union[UnitaryM, CtrlGate]], dict]:
        meta = dict()

        if grain < EmitType.TWO_LEVEL:
            result = mat2l_decompose(u)
            meta['method'] = 'mat2l_decompose'
        else:
            assert u.is2l()
            result = cnot_decompose(u)
            meta['method'] = 'cnot_decompose'
        return result, meta

    def finish(self, optimized=False) -> object:
        return self.builder.finish(optimized=optimized)

    def all_qubits(self):
        return self.builder.all_qubits()
