from enum import Enum, IntFlag, IntEnum
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from quompiler.utils.mfun import allprop


class UnivGate(Enum):
    I = ('I', np.eye(2))
    X = ('X', np.eye(2)[[1, 0]])
    Y = ('Y', np.array([[0j, -1j], [1j, 0j]]))
    Z = ('Z', np.array([[1, 0j], [0j, -1]]))
    H = ('H', np.array([[1, 1], [1, -1]]) / np.sqrt(2))
    S = ('S', np.array([[1, 0j], [0j, 1j]]))
    T = ('T', np.array([[1, 0j], [0j, np.exp(1j * np.pi / 4)]]))
    SD = ('SD', np.array([[1, 0j], [0j, -1j]]))
    TD = ('TD', np.array([[1, 0j], [0j, np.exp(-1j * np.pi / 4)]]))

    def __init__(self, label, mat: NDArray):
        self.label = label
        self.matrix = mat

    @staticmethod
    def get(m: NDArray) -> Optional['UnivGate']:
        """
        Attempt to match exactly a universal gate within certain tolerance.
        :param m:
        :return: the matching universal gate. None if no universal gate matches.
        """
        if m.shape != (2, 2) or not np.allclose(m.conj() @ m.T, np.eye(2)):
            return None
        for g in UnivGate:
            if np.allclose(m, g.matrix):
                return g
        return None

    def herm(self):
        if self == UnivGate.S:
            return UnivGate.SD
        if self == UnivGate.SD:
            return UnivGate.S
        if self == UnivGate.T:
            return UnivGate.TD
        if self == UnivGate.TD:
            return UnivGate.T
        return self

    @staticmethod
    def get_prop(m: NDArray) -> Optional[tuple['UnivGate', complex]]:
        """
        Attempt to match a universal gate within certain tolerance other than a phase factor (of norm 1), e.g., m = λg with λ as a phase factor.
        :param m:
        :return: the matching universal gate and the phase factor, if any. None if no universal gate matches.
        """
        if m.shape != (2, 2) or not np.allclose(m.conj() @ m.T, np.eye(2)):
            return None
        for g in UnivGate:
            if np.allclose(m, g.matrix):
                return g, 1
            p, f = allprop(m, g.matrix)
            if p:
                return g, f
        return None

    def rotation(self, theta) -> NDArray:
        """
        Calculate the rotation matrix corresponding to the given angle.
        :param theta: the angle in radians.
        :return: the rotation matrix corresponding to the given angle along the axis.
        """
        return expm(-.5j * theta * self.matrix)

    @staticmethod
    def cliffordt():
        return [UnivGate.I, UnivGate.X, UnivGate.H, UnivGate.S, UnivGate.T, UnivGate.SD, UnivGate.TD]


class QType(IntFlag):
    """
    This is a classification of the qubit type in quantum computing unitary transformations.
    There are a few predefined combinations:
    EXTENSION = IDLER | TARGET, is the extended matrix combining IDLER bits and TARGET bits
    CONTROL = CONTROL0 | CONTROL1, is the control bits
    """
    TARGET = (1, (0, 1))  # target qubit
    CONTROL0 = (2, (0,))  # control qubit with activation value 0 (type 0 control)
    CONTROL1 = (4, (1,))  # control qubit with value activation 1 (type 1 control)
    IDLER = (8, (0, 1))  # the non-interactive bystander. they are neither target nor control.

    def __new__(cls, value, base):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.base = base  # the base to be spanned by this QType
        return obj

    def __repr__(self):
        if self.name:
            return self.name
        return repr(int(self))


class EmitType(IntEnum):
    """
    This enum specifies the granularity level of unitary operator.

    The granularity is provided in terms as follows:
    """
    INVALID = 0
    UNITARY = 0x12  # any n-order UnitaryM
    TWO_LEVEL = 0x16  # any UnitaryM with no more than 2 non-identity rows/cols
    MULTI_TARGET = 0x22,  # CtrlGate and with more than one qubit in target
    SINGLET = 0x26,  # CtrlGate and with one-qubit target
    CTRL_PRUNED = 0x2a,  # CtrlGate with single target and zero or one control qubit.
    UNIV_GATE = 0x32,  # any std CtrlGate with any UnivGate as the operator
    CLIFFORD_T = 0x38,  # CtrlGate and the gate is among Clifford + T gates, namely, {X, H, S, T}

    def __repr__(self):
        return self.name


class QompilePlatform(Enum):
    CIRQ = 'CirqBuilder'
    QISKIT = 'QiskitBuilder'
    QUIMB = 'QuimbBuilder'

    def __repr__(self):
        return self.name


class OptLevel(Enum):
    """
    Optimization level for compiler.
    """
    O0 = 'O0'  # basic optimization
    O1 = 'O1'  # mild optimization
    O2 = 'O2'  # advanced optimization
    O3 = 'O3'  # bold optimization
