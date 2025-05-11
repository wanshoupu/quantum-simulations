from enum import Enum, IntFlag, IntEnum
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm


class UnivGate(Enum):
    I = ('I', np.eye(2))
    X = ('X', np.eye(2)[[1, 0]])
    Y = ('Y', np.array([[0j, -1j], [1j, 0j]]))
    Z = ('Z', np.array([[1, 0j], [0j, -1]]))
    H = ('H', np.array([[1, 1], [1, -1]]) / np.sqrt(2))
    S = ('S', np.array([[1, 0j], [0j, 1j]]))
    T = ('T', np.array([[1, 0j], [0j, np.exp(1j * np.pi / 4)]]))

    def __init__(self, label, mat: NDArray):
        self.label = label
        self.matrix = mat

    @staticmethod
    def get(m: NDArray) -> Optional['UnivGate']:
        """
        Attempt to match to a universal gate.
        :param m:
        :return: the matching universal gate. None if no universal gate matches.
        """
        for g in UnivGate:
            if m.shape == g.matrix.shape and np.allclose(m, g.matrix):
                return g
        return None

    def rotation(self, theta) -> NDArray:
        """
        Calculate the rotation matrix corresponding to the given angle.
        :param theta: the angle in radians.
        :return: the rotation matrix corresponding to the given angle along the axis.
        """
        return expm(-.5j * theta * self.matrix)


class QType(IntFlag):
    """
    This is a classification of the qubit type in quantum computing unitary transformations.
    There are a few predefined combinations:
    EXTENSION = IDLER | TARGET, is the extended matrix combining IDLER bits and TARGET bits
    CONTROL = CONTROL0 | CONTROL1, is the control bits
    """
    IDLER = (1, (0, 1))  # the non-interactive bystander. they are neither target nor control.
    TARGET = (2, (0, 1))  # target qubit
    CONTROL0 = (4, (0,))  # control qubit with activation value 0 (type 0 control)
    CONTROL1 = (8, (1,))  # control qubit with value activation 1 (type 1 control)

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
    UNITARY = 0x12  # any n-order UnitaryM with more than 2 non-identity rows/cols
    TWO_LEVEL = 0x16  # any UnitaryM with no more than 2 non-identity rows/cols
    MULTI_TARGET = 0x22,  # CtrlGate and with more than one qubit in target
    SINGLET = 0x26,  # CtrlGate and with no more than one qubit in target
    TWO_CTRL = 0x2a,  # CtrlGate and with no more than two qubit in control sequence
    ONE_CTRL = 0x2c,  # CtrlGate and with no more than one qubit in control sequence
    UNIV_GATE = 0x32,  # any CtrlStdGate
    CLIFFORD_T = 0x38,  # CtrlStdGate and the gate is among Clifford + T gates, namely, {X, H, S, T}

    def __repr__(self):
        return self.name


class QompilePlatform(Enum):
    CIRQ = 'CirqBuilder'
    QISKIT = 'QiskitBuilder'
    QUIMB = 'QuimbBuilder'

    def __repr__(self):
        return self.name
