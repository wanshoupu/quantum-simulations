"""
This module provide the compilation functionalities.
If needed, it may make distinctions between target qubits and ancilla qubits.
"""
from typing import Union

from numpy._typing import NDArray

from quompiler.circuits.cirq_circuit import CirqBuilder
from quompiler.circuits.qiskit_circuit import QiskitBuilder
from quompiler.circuits.quimb_circuit import QuimbBuilder
from quompiler.construct.bytecode import BytecodeRevIter, Bytecode
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.qspace import Ancilla
from quompiler.construct.std_gate import CtrlStdGate
from quompiler.construct.types import QompilePlatform, EmitType, UnivGate
from quompiler.construct.unitary import UnitaryM
from quompiler.qompile.configure import QompilerConfig
from quompiler.utils.granularity import granularity
from quompiler.utils.cnot_decompose import cnot_decompose
from quompiler.utils.mat2l_decompose import mat2l_decompose
from quompiler.utils.std_decompose import std_decompose, ctrl_decompose


class Qompiler:

    def __init__(self, config: QompilerConfig):
        self.config = config
        self.builder = self.create_builder(QompilePlatform[config.target])
        self.emit = EmitType[config.emit]
        self.aspace = [Ancilla(i) for i in range(*config.device.arange)]

    def interpret(self, u: NDArray):
        component = self.compile(u)
        for c in BytecodeRevIter(component):
            m = c.data
            if isinstance(m, CtrlStdGate) or isinstance(m, CtrlGate):
                self.builder.build_gate(m)
            elif isinstance(m, UnitaryM):
                self.builder.build_group(m)

    def compile(self, u: NDArray) -> Bytecode:
        s = u.shape
        um = UnitaryM(s[0], tuple(range(s[0])), u)
        return self._decompose(um)

    def _decompose(self, data: Union[UnitaryM, CtrlGate, CtrlStdGate]) -> Bytecode:
        root = Bytecode(data)
        g = granularity(data)
        if self.emit <= g:  # noop
            return root

        if isinstance(data, UnitaryM):
            constituents = self._decompose_unitary(g, data)
        elif isinstance(data, CtrlGate):
            constituents = self._decompose_ctrl(g, data)
        elif isinstance(data, CtrlStdGate):
            constituents = self._decompose_std(data)
        else:
            raise ValueError(f"Unrecognized gate of type {type(g)}")
        # decompose is noop
        if len(constituents) == 1 and constituents[0] == data:
            return root
        for c in constituents:
            root.append(self._decompose(c))
        return root

    def _decompose_std(self, u):
        std_gates = UnivGate.cliffordt() if self.emit == EmitType.CLIFFORD_T else list(UnivGate)
        constituents = std_decompose(u, std_gates, self.config.rtol, self.config.atol)
        return constituents

    def _decompose_ctrl(self, grain, gate):
        # EmitType.MULTI_TARGET is disabled atm
        # if g < EmitType.MULTI_TARGET:
        #     result = ctrl_decompose(u, clength=2, aspace=self.aspace)
        if grain < EmitType.CTRL_PRUNED:
            result = ctrl_decompose(gate, clength=1, aspace=self.aspace)
        else:
            result = self._decompose_std(gate)
        return result

    @staticmethod
    def _decompose_unitary(g, u):
        if g < EmitType.TWO_LEVEL:
            result = mat2l_decompose(u)
        else:
            assert u.is2l()
            result = cnot_decompose(u)
        return result

    def finish(self, optimized=False) -> object:
        return self.builder.finish(optimized=optimized)

    def all_qubits(self):
        return self.builder.all_qubits()

    def create_builder(self, platform: QompilePlatform):
        if platform == QompilePlatform.CIRQ:
            return CirqBuilder(self.config.device)
        if platform == QompilePlatform.QISKIT:
            return QiskitBuilder(self.config.device)
        if platform == QompilePlatform.QUIMB:
            return QuimbBuilder(self.config.device)
        raise NotImplementedError(f"Unsupported platform: {platform}")
