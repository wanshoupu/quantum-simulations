from typing import Union, Sequence

import numpy as np
from numpy.typing import NDArray

from quompiler.construct.types import UnivGate
from quompiler.utils.fun import rational, pi_repr
from quompiler.utils.su2fun import rot, vec

_principal_axes = {"x": np.array([1, 0, 0]), "y": np.array([0, 1, 0]), "z": np.array([0, 0, 1]),
                   "-x": np.array([-1, 0, 0]), "-y": np.array([0, -1, 0]), "-z": np.array([0, 0, -1])}


class RAxis:
    """
    Represent the rotation axis with a few convenient ways to define.
    """

    def __init__(self, axis: Union[str, np.ndarray, Sequence]):
        """
        Normal vector, n, may be specified in one of three ways:
        3-vector: (x,y,z) represents the normal vector of the rotation axis. It can be an unnormalized vector as normalization will be performed internally.
        2-vector: (θ,φ) represents the normal vector of the rotation axis in spherical coordinate.
        :param axis: axis of the rotation matrix
            when it's a string, it must be in {'x', 'y', 'z', '-x', '-y', '-z'};
            when it's a 2-vector, it is interpreted as spherical coordinate;
            when it's a 3-vector, it is interpreted as normal vector (x,y,z).
        """
        if isinstance(axis, str):
            if axis not in _principal_axes:
                raise ValueError("axis must be in {'x', 'y', 'z', '-x', '-y', '-z'}")
            self.nvec = _principal_axes[axis]
        elif isinstance(axis, np.ndarray) or isinstance(axis, Sequence):
            axis = np.array(axis)
            if axis.shape == (2,):
                theta, phi = axis
                self.nvec = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
            elif axis.shape == (3,):
                norm = np.linalg.norm(axis)
                if np.isclose(norm, 0):
                    raise ValueError("axis must not be zero.")
                self.nvec = axis / norm
            else:
                raise ValueError('axis must be either a 2-vector or 3-vector')
        else:
            raise ValueError('axis must be either an np.array or a str or RAxis')
        x, y, z = self.nvec
        self.theta = np.arccos(np.clip(z, -1, 1))
        self.phi = np.arctan2(y, x)

        self.principal = None
        for key, val in _principal_axes.items():
            if np.allclose(self.nvec, val):
                self.principal = key

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, RAxis):
            return False
        if self.isprincipal() and other.isprincipal() and self.principal == other.principal:
            return True
        return np.allclose(self.nvec, other.nvec)

    def __repr__(self):
        if self.isprincipal():
            return self.principal
        return f"({pi_repr(self.theta)}, {pi_repr(self.phi)})"

    def spherical(self) -> tuple[float, float]:
        """
        Get the spherical coordinate tuple(θ, φ)
        """
        return self.theta, self.phi

    def isprincipal(self):
        return bool(self.principal)


class RGate:
    """
    Represents a SU2 rotation matrix for a single qubit, parameterized as angle and axis.
        U = cos(α/2) I - i sin(α/2) σ n
    where n is the normal vector, α is the angle of rotation.
    """

    def __init__(self, angle: float, axis: Union[str, RAxis, np.ndarray, Sequence]):
        """
        Create a rotation matrix from an axis and angle.
        Normal vector, n, may be specified in one of three ways:
        3-vector: (x,y,z) represents the normal vector of the rotation axis. It can be an unnormalized vector as normalization will be performed internally.
        2-vector: (θ,φ) represents the normal vector of the rotation axis in spherical coordinate.
        :param angle: angle of the rotation matrix
        :param axis: axis of the rotation matrix
            when it's a string, it can only be one of 'X' or 'Y' or 'Z';
            when it's a 2-vector, it is interpreted as spherical coordinate;
            when it's a 3-vector, it is interpreted as normal vector (x,y,z).
        """
        if isinstance(axis, RAxis):
            self.axis = axis
        elif isinstance(axis, str) or isinstance(axis, np.ndarray) or isinstance(axis, Sequence):
            self.axis = RAxis(axis)
        else:
            raise ValueError('axis must be either an np.array or a str or RAxis')
        self.angle = angle
        self.matrix = rot(self.axis.nvec, self.angle)
        self.gate = UnivGate.get(self.matrix)

    def __repr__(self):
        if self.gate:
            return repr(self.gate)
        return f"R{repr(self.axis)}({pi_repr(self.angle)})"

    def __matmul__(self, other: Union[NDArray, 'RGate']) -> Union[NDArray, 'RGate']:
        """
        Calculate the multiplication between this RMat and another RMat and return a new RMat,
        R = self @ other, other than a global phase.
        :param other:
        :return: a new RMat such that R = self @ other, other than a global phase.
        """
        if isinstance(other, np.ndarray):
            return np.array(self) @ other
        if not isinstance(other, RGate):
            raise NotImplementedError(f'matmul only implemented for {other}')
        if self.axis == other.axis:
            return RGate(self.angle + other.angle, self.axis)
        mat = self.matrix @ other.matrix
        theta, phi, alpha = vec(mat)
        return RGate(alpha, np.array([theta, phi]))

    def __array__(self) -> NDArray:
        return self.matrix
