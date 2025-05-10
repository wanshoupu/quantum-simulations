"""
This module provide the compilation functionalities.
If needed, it may make distinctions between target qubits and ancilla qubits.
"""
from typing import Union, Sequence

from numpy.typing import NDArray

from quompiler.construct.bytecode import Bytecode, ReverseBytecodeIter
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.qspace import Ancilla
from quompiler.construct.std_gate import CtrlStdGate
from quompiler.construct.types import UnivGate
from quompiler.construct.unitary import UnitaryM
from quompiler.qompile.configure import QompilerConfig, QompilePlatform, EmitType
from quompiler.utils.cnot_decompose import cnot_decompose
from quompiler.utils.mat2l_decompose import mat2l_decompose
from quompiler.utils.mat_utils import validm2l
from quompiler.utils.std_decompose import ctrl_decompose, std_decompose


class Qompiler:

    def __init__(self, config: QompilerConfig):
        self.config = config
        dim = config.device.dimension
        self.builder = QompilePlatform[config.target](dim)
        self.emit = EmitType[config.emit]
        self.aspace = config.device.aspace

    def interpret(self, u: NDArray):
        component = self.compile(u)
        for c in ReverseBytecodeIter(component):
            m = c.data
            if isinstance(m, CtrlStdGate) or isinstance(m, CtrlGate):
                # TODO for now draw single-qubit + controlled single-qubit as gate.
                # TO BE breakdown further to elementary gates only
                self.builder.build_gate(m)
            elif isinstance(m, UnitaryM):
                self.builder.build_group(m)

    def compile(self, u: NDArray) -> Bytecode:
        s = u.shape
        um = UnitaryM(s[0], tuple(range(s[0])), u)
        return self._compile(um)

    def _compile(self, u: Union[UnitaryM, CtrlGate, CtrlStdGate]) -> Bytecode:
        if isinstance(u, UnitaryM):
            coms = self._decompose_unitary(u)
        elif isinstance(u, CtrlGate):
            coms = self._decompose_ctrgate(u)
        else:
            coms = [u]
        # creates Bytecode
        if len(coms) == 1:
            return Bytecode(coms[0])
        root = Bytecode(u)
        for c in coms:
            child = self._compile(c)
            root.append(child)
        return root

    def _decompose_unitary(self, u: UnitaryM) -> Sequence[Union[CtrlGate, UnitaryM]]:
        if EmitType.UNITARY < self.emit:
            return mat2l_decompose(u)

        if EmitType.TWO_LEVEL < self.emit and u.is2l():
            return cnot_decompose(u)

        return [u]

    def _decompose(self, u: Union[UnitaryM, CtrlGate, CtrlStdGate]):

        if isinstance(u, UnitaryM):
            mat = u.matrix
        else:
            raise TypeError("Unsupported unitary matrix")
        if validm2l(mat):
            return cnot_decompose(u)
        return mat2l_decompose(u)

    def finish(self, optimized=False) -> object:
        return self.builder.finish(optimized=optimized)

    def all_qubits(self):
        return self.builder.all_qubits()

    def _decompose_ctrgate(self, gate: CtrlGate) -> Sequence[Union[CtrlGate, CtrlStdGate]]:
        if EmitType.TOFFOLI < self.emit and 1 < len(gate.control_qids()):
            clength = 2 if self.emit == EmitType.TOFFOLI else 1
            return ctrl_decompose(gate, clength=clength, aspace=self.config.device.aspace)

        if EmitType.UNIV_GATE <= self.emit:
            ugs = list(UnivGate) if self.emit == EmitType.UNIV_GATE else [UnivGate.X, UnivGate.H, UnivGate.S, UnivGate.T]
            return std_decompose(gate, ugs, rtol=self.config.rtol, atol=self.config.atol)
        return [gate]
