"""
This module provide the compilation functionalities.
If needed, it may make distinctions between target qubits and ancilla qubits.
"""
from typing import Union

from numpy.typing import NDArray

from quompiler.circuits.cirq_circuit import CirqBuilder
from quompiler.circuits.qiskit_circuit import QiskitBuilder
from quompiler.circuits.quimb_circuit import QuimbBuilder
from quompiler.construct.bytecode import Bytecode, ReverseBytecodeIter
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.qspace import Ancilla
from quompiler.construct.std_gate import CtrlStdGate
from quompiler.construct.types import UnivGate, EmitType, QompilePlatform
from quompiler.construct.unitary import UnitaryM
from quompiler.qompile.configure import QompilerConfig
from quompiler.utils.cnot_decompose import cnot_decompose
from quompiler.utils.mat2l_decompose import mat2l_decompose
from quompiler.utils.std_decompose import ctrl_decompose, std_decompose


def granularity(obj: Union[UnitaryM, CtrlGate, CtrlStdGate]) -> EmitType:
    """
    Given an input obj, determine the granularity level if we are to return it as is.
    The granularity is provided in terms of EmitType (see quompiler.construct.types.EmitType).
    :param obj: input object of one of the types
    :return: EmitType denoting the granularity
    """
    if isinstance(obj, CtrlStdGate):
        if obj.gate in UnivGate.cliffordt():
            return EmitType.CLIFFORD_T
        return EmitType.UNIV_GATE

    if isinstance(obj, CtrlGate):
        if len(obj.control_qids()) == 1:
            return EmitType.ONE_CTRL
        if len(obj.control_qids()) == 2:
            return EmitType.TWO_CTRL
        if obj.issinglet():
            return EmitType.SINGLET
        return EmitType.MULTI_TARGET

    if isinstance(obj, UnitaryM):
        if obj.is2l():
            return EmitType.TWO_LEVEL
        return EmitType.UNITARY
    return EmitType.INVALID


class Qompiler:

    def __init__(self, config: QompilerConfig):
        self.config = config
        self.builder = self.create_builder(QompilePlatform[config.target])
        self.emit = EmitType[config.emit]
        self.aspace = [Ancilla(i) for i in range(*config.device.arange)]

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
        root = Bytecode(um)
        self._decompose(root)
        return root

    def _decompose(self, root: Bytecode) -> None:
        g = granularity(root.data)
        if self.emit <= g:  # noop
            return

        u = root.data
        if isinstance(u, UnitaryM):
            if g < EmitType.TWO_LEVEL:
                coms = mat2l_decompose(u)
            else:  # g < EmitType.SINGLET
                coms = cnot_decompose(u)
        elif isinstance(u, CtrlGate):
            if g < EmitType.UNIV_GATE:
                coms = std_decompose(u, list(UnivGate), self.config.rtol, self.config.atol)
            elif g < EmitType.ONE_CTRL:
                coms = ctrl_decompose(u, clength=1, aspace=self.aspace)
            elif g < EmitType.TWO_CTRL:
                coms = ctrl_decompose(u, clength=2, aspace=self.aspace)
            elif g < EmitType.MULTI_TARGET:  # this should be impossible
                coms = ctrl_decompose(u, clength=2, aspace=self.aspace)
            else:
                coms = std_decompose(u, UnivGate.cliffordt(), self.config.rtol, self.config.atol)
        elif isinstance(u, CtrlStdGate):
            if g < EmitType.CLIFFORD_T:
                coms = std_decompose(u, UnivGate.cliffordt(), self.config.rtol, self.config.atol)
            else:  # already at the finest granularity, CLIFFORD_T
                coms = [u]
        else:
            raise ValueError(f"Unrecognized gate of type {type(g)}")
        # decompose is noop
        if len(coms) == 1:
            return
        for c in coms:
            root.append(Bytecode(c))

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
