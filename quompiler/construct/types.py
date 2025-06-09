from enum import Enum, IntFlag, IntEnum
from typing import Optional, Union, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from quompiler.utils.mfun import allprop, ConditionalResult

_univgate_mat_mapping = dict([
    ('I', np.eye(2)),
    ('X', np.eye(2)[[1, 0]]),
    ('Y', np.array([[0j, -1j], [1j, 0j]])),
    ('Z', np.array([[1, 0j], [0j, -1]])),
    ('H', np.array([[1, 1], [1, -1]]) / np.sqrt(2)),
    ('S', np.array([[1, 0j], [0j, 1j]])),
    ('T', np.array([[1, 0j], [0j, np.exp(1j * np.pi / 4)]])),
    ('SD', np.array([[1, 0j], [0j, -1j]])),
    ('TD', np.array([[1, 0j], [0j, np.exp(-1j * np.pi / 4)]])),
])


class UnivGate(Enum):
    I = 'I'
    X = 'X'
    Y = 'Y'
    Z = 'Z'
    H = 'H'
    S = 'S'
    T = 'T'
    SD = 'SD'
    TD = 'TD'

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
            if np.allclose(m, np.array(g)):
                return g
        return None

    def __repr__(self):
        return self.name

    def __lt__(self, other):
        return self.name < other.name

    def __reduce__(self):
        return self.__class__, (self.name,)

    def __array__(self):
        return _univgate_mat_mapping[self.value]

    def __matmul__(self, other: Union[NDArray, 'UnivGate']) -> NDArray:
        # 'other' may be np.ndarray
        return np.array(self) @ other

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
    def get_prop(m: NDArray) -> Optional['UnivGate']:
        """
        Attempt to match a universal gate within certain tolerance other than a phase factor (of norm 1), e.g., m = λg with λ as a phase factor.
        :param m:
        :return: the matching universal gate and the phase factor, if any. None if no universal gate matches.
        """
        if m.shape != (2, 2) or not np.allclose(m.conj() @ m.T, np.eye(2)):
            return None
        for g in UnivGate:
            if np.allclose(m, np.array(g)) or allprop(m, np.array(g)):
                return g
        return None

    def rotation(self, theta) -> NDArray:
        """
        Calculate the rotation matrix corresponding to the given angle.
        :param theta: the angle in radians.
        :return: the rotation matrix corresponding to the given angle along the axis.
        """
        return expm(-.5j * theta * np.array(self))

    @staticmethod
    def cliffordt():
        return [UnivGate.I, UnivGate.X, UnivGate.H, UnivGate.Z, UnivGate.S, UnivGate.T, UnivGate.SD, UnivGate.TD]


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
    UNITARY = 0x20  # any n-order UnitaryM
    TWO_LEVEL = 0x40  # any UnitaryM with no more than 2 non-identity rows/cols
    MULTI_TARGET = 0x60,  # CtrlGate and with more than one qubit in target
    SINGLET = 0x80,  # CtrlGate and with one-qubit target
    CTRL_PRUNED = 0xa0,  # CtrlGate with single target and zero or one control qubit.
    PRINCIPAL = 0xb0,  # All single qubit operations are Rx(α), Ry(α), or Rz(α) with rotation angles 0 <= α < 2π.
    UNIV_GATE = 0xc0,  # any std CtrlGate with any UnivGate as the operator
    CLIFFORD_T = 0xe0,  # CtrlGate and the gate is among Clifford + T gates, namely, {X, H, S, T}

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


class SU2NetType(Enum):
    """
    Implementations of SU2Net.
    """
    BruteNN = 1
    AutoNN = 2
    BallTreeNN = 3
    KDTreeNN = 4


class PrincipalAxis(Enum):
    X = (1, 0, 0)
    Y = (0, 1, 0)
    Z = (0, 0, 1)

    def __repr__(self):
        return self.name.lower()

    @staticmethod
    def get(axis: Union[NDArray, Sequence]) -> Optional['PrincipalAxis']:
        axis = np.array(axis)
        length, = axis.shape
        assert length == 2 or length == 3

        if length == 3:
            for k in PrincipalAxis:
                if np.allclose(axis, np.array(k.value)):
                    return k
            return None

        vec = np.sin(axis[0]) * np.cos(axis[1]), np.sin(axis[0]) * np.sin(axis[1]), np.cos(axis[0])
        return PrincipalAxis.get(vec)

    @staticmethod
    def get_prop(axis: Union[NDArray, Sequence]) -> ConditionalResult:
        """
        Check if `axis` represents one of the principal axes: 'x', 'y', 'z'.
        If so, return a tuple of the principal axis and a float factor to denote if its parallel (positive number) or antiparallel (negative number).
        :param axis: given in 3D/2D vector, corresponding to Euclidean vector or spherical vector.
        :return: ConditionalResult representing the check.
        """
        axis = np.array(axis)
        length, = axis.shape
        assert length == 2 or length == 3

        if length == 3:
            for k in PrincipalAxis:
                pchk = allprop(axis, np.array(k.value))
                if pchk:
                    result = k, pchk.result
                    return ConditionalResult(True, result)
            return ConditionalResult()

        vec = np.sin(axis[0]) * np.cos(axis[1]), np.sin(axis[0]) * np.sin(axis[1]), np.cos(axis[0])
        return PrincipalAxis.get_prop(vec)
