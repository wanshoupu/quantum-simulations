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
from quompiler.construct.solovay import SKDecomposer
from quompiler.construct.types import EmitType, UnivGate
from quompiler.construct.unitary import UnitaryM
from quompiler.optimize.optimizer import Optimizer
from quompiler.utils.cnot_decompose import cnot_decompose
from quompiler.utils.ctrl_decompose import ctrl_decompose
from quompiler.utils.euler_decompose import euler_decompose
from quompiler.utils.granularity import granularity
from quompiler.utils.mat2l_decompose import mat2l_decompose
from quompiler.utils.std_decompose import cliffordt_decompose


class Qompiler:

    def __init__(self, config: QompilerConfig, builder: CircuitBuilder, device: QDevice, optimizers: list[Optimizer] = None):
        self.config = config
        self.builder = builder
        self.device = device
        self.optimizers = optimizers or []
        self.emit = EmitType[config.emit]
        self.debug = self.config.debug
        self.sk = SKDecomposer(config.rtol, config.atol)

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
        if self.debug:
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
            constituents, meta = self._decompose_ctrlgate(grain, data)
        else:
            raise ValueError(f"Unrecognized gate of type {type(grain)}")
        if self.debug:
            root.metadata.update(meta)

        # decompose is noop
        if len(constituents) == 1 and constituents[0] == data:
            return root
        if self.debug:
            root.metadata['fanout'] = len(constituents)
        for c in constituents:
            root.add_child(self._decompose(c))
        return root

    def _decompose_std(self, gate: CtrlGate) -> tuple[list[CtrlGate], dict]:
        if gate.is_std():
            if self.emit == EmitType.UNIV_GATE or gate.gate in UnivGate.cliffordt():
                return [gate], {}
            else:
                meta = {'method': 'cliffordt_decompose'} if self.debug else {}
                return cliffordt_decompose(gate), meta
        sk_coms = self.sk.approx(gate.matrix())
        constituents = [CtrlGate(g, gate.controls, gate.qspace) for g in sk_coms]
        meta = {'method': 'sk_approx'} if self.debug else {}
        return constituents, meta

    def _decompose_ctrlgate(self, grain: EmitType, gate: CtrlGate) -> tuple[list[CtrlGate], dict]:
        # EmitType.MULTI_TARGET is disabled atm
        # if g < EmitType.MULTI_TARGET:
        #     result = ctrl_decompose(u, clength=2, aspace=self.aspace)
        if grain < EmitType.CTRL_PRUNED:
            result = ctrl_decompose(gate, self.device, clength=1)
            meta = {'method': 'ctrl_decompose'} if self.debug else {}
            return result, meta

        euler_coms = euler_decompose(gate)
        result = []
        meta = dict()
        for com in euler_coms:
            std_coms, meta1 = self._decompose_std(com)
            result.extend(std_coms)
            if self.debug:
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
