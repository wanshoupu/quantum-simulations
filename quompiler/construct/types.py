from enum import Enum, IntFlag
from typing import Optional

import numpy as np
from numpy.typing import NDArray


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
        self.mat = mat

    @staticmethod
    def get(m: NDArray) -> Optional['UnivGate']:
        for g in UnivGate:
            if m.shape == g.mat.shape and np.allclose(m, g.mat):
                return g


class QType(IntFlag):
    """
    This is a classification of the qubit type in quantum computing unitary transformations.
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
        return self.name
