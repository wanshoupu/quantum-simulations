from enum import Enum
from typing import Union, Optional

import numpy as np
from numpy.typing import NDArray

from quompiler.construct.qspace import Qubit
from quompiler.construct.types import UnivGate
from quompiler.utils.su2fun import rot


class RotAxis(Enum):
    X = (1, 0, 0)
    Y = (0, 1, 0)
    Z = (0, 0, 1)

    @staticmethod
    def get(m: NDArray) -> Optional['RotAxis']:
        """
        Attempt to match to one of the axis X, Y, or Z within certain tolerance.
        :param m:
        :return: the matching universal gate. None if no universal gate matches.
        """
        if m.shape != (3,):
            return None
        for g in RotAxis:
            if np.allclose(m, g.value):
                return g
        return None


class RGate:
    """
    Represents a rotation matrix for a single qubit, parameterized as angle and axis.
        U = cos(α/2) I - i sin(α/2) σ n
    where n is the normal vector, α is the angle of rotation.
    """

    def __init__(self, axis: Union[str, np.ndarray], angle: float, qubit: Qubit):
        """
        Create a rotation matrix from an axis and angle and phase.
        Normal vector, n, may be specified in one of three ways:
        3-vector: (x,y,z) represents the normal vector of the rotation axis. It can be an unnormalized vector as normalization will be performed internally.
        2-vector: (θ,φ) represents the normal vector of the rotation axis in spherical coordinate.
        :param axis: axis of the rotation matrix
            when it's a string, it can only be one of 'X' or 'Y' or 'Z';
            when it's a 2-vector, it is interpreted as spherical coordinate;
            when it's a 3-vector, it is interpreted as normal vector (x,y,z).
        :param angle: angle of the rotation matrix
        :param qubit: the qubit space this operator will operate on.
        """
        if isinstance(axis, np.ndarray):
            if axis.shape == (2,):
                theta, phi = axis
                self.axis = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
            elif axis.shape == (3,):
                self.axis = axis / np.linalg.norm(axis)
            else:
                raise ValueError('axis must be either a 2-vector or 3-vector')
            self.aligned_axis = RotAxis.get(self.axis)
        elif isinstance(axis, str):
            self.aligned_axis = RotAxis[axis]
            self.axis = np.array(self.aligned_axis.value)
        else:
            raise ValueError('axis must be either an np.array or a str')
        self.angle = angle
        self.matrix = rot(self.axis, self.angle)
        self.gate = UnivGate.get(self.matrix)
        self.qubit = qubit

    def __repr__(self):
        if self.gate:
            return repr(self.gate)
        if isinstance(self.axis, str):
            return f"R{self.axis}({self.angle})"
        return f"R({self.axis}, {self.angle})"

    def __matmul__(self, other: "RGate") -> "RGate":
        if self.axis == other.axis:
            return RGate(self.axis, self.angle + other.angle, self.qubit)
        raise NotImplementedError(f"Cannot perform {self} and {other}")

    def __array__(self) -> NDArray:
        return self.matrix

    def inflate(self) -> NDArray:
        return self.matrix

    def is_axis_aligned(self) -> bool:
        return bool(self.aligned_axis)
