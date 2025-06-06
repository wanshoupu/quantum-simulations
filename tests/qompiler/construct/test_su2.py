from typing import Union

import numpy as np
import pytest

from quompiler.construct.qspace import Qubit
from quompiler.construct.su2gate import RGate, RAxis, _principal_axes
import random


@pytest.mark.parametrize("axis, error", [
    ["h", "axis must be in {'x', 'y', 'z', '-x', '-y', '-z'}"],
    ["-t", "axis must be in {'x', 'y', 'z', '-x', '-y', '-z'}"],
    ["-s", "axis must be in {'x', 'y', 'z', '-x', '-y', '-z'}"],
    [np.array([1]), 'axis must be either a 2-vector or 3-vector'],
    [np.array([1, 2, 3, 4]), 'axis must be either a 2-vector or 3-vector'],
    [np.array([0, 0, 0]), "axis must not be zero."],
])
def test_axis_init_invalid(axis: Union[str, np.ndarray], error):
    with pytest.raises(ValueError) as e:
        RAxis(axis)
    assert str(e.value) == error


@pytest.mark.parametrize("principal", ["x", "y", "z", "-x", "-y", "-z"])
def test_axis_init_principal(principal):
    axis = RAxis(principal)
    assert repr(axis) == principal
    assert np.allclose(axis.nvec, _principal_axes[principal])


@pytest.mark.parametrize("spherical, expected", [
    [[0, 0], [0, 0]],
    [[np.pi, 0], [np.pi, 0]],
    [[0, 2 * np.pi], [0, 0]],
    [[0, -2 * np.pi], [0, 0]],
    [[-np.pi / 2, 0], [np.pi / 2, -np.pi]],
    [[-np.pi / 2, np.pi / 2], [np.pi / 2, -np.pi / 2]],
    [[-np.pi / 3, np.pi / 4], [np.pi / 3, -3 * np.pi / 4]],
])
def test_axis_init_spherical(spherical: list, expected):
    axis = RAxis(np.array(spherical))
    assert np.allclose(axis.spherical(), expected)


@pytest.mark.parametrize("spherical, expected", [
    [[0, 0], [0, 0, 1]],
    [[np.pi, 0], [0, 0, -1]],
    [[0, 2 * np.pi], [0, 0, 1]],
    [[0, -2 * np.pi], [0, 0, 1]],
    [[-np.pi / 2, 0], [-1, 0, 0]],
    [[-np.pi / 2, np.pi / 2], [0, -1, 0]],
    [[-np.pi / 3, np.pi / 4], [-np.sqrt(3) / 2 / np.sqrt(2), -np.sqrt(3) / 2 / np.sqrt(2), .5]],
])
def test_axis_init_spherical2nvec(spherical: list, expected):
    axis = RAxis(np.array(spherical))
    assert np.allclose(axis.nvec, expected), f"{axis.nvec} != {expected}"


@pytest.mark.parametrize("nvec, expected", [
    [[1e-5, 0, 0], [1, 0, 0]],
    [[1, 0, 0], [1, 0, 0]],
    [[0, 1, 0], [0, 1, 0]],
    [[0, 0, 1], [0, 0, 1]],
    [[0, .5, 0], [0, 1, 0]],
    [[0, 0, -1], [0, 0, -1]],
    [[0, 0, -.5], [0, 0, -1]],
    [[-1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)], [-1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)]],
])
def test_axis_init_nvec(nvec, expected):
    axis = RAxis(np.array(nvec))
    assert np.allclose(axis.nvec, expected), f"{axis.nvec} != {expected}"


@pytest.mark.parametrize("nvec, expected", [
    [[1e-5, 0, 0], [np.pi / 2, 0]],
    [[1, 0, 0], [np.pi / 2, 0]],
    [[0, 1, 0], [np.pi / 2, np.pi / 2]],
    [[0, 0, 1], [0, 0]],
    [[0, .5, 0], [np.pi / 2, np.pi / 2]],
    [[0, 0, -1], [np.pi, 0]],
    [[0, 0, -.5], [np.pi, 0]],
    [[-1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)], [np.arcsin(np.sqrt(2 / 3)), 3 * np.pi / 4]],
])
def test_axis_init_nvec2spherical(nvec, expected):
    axis = RAxis(np.array(nvec))
    assert np.allclose(axis.spherical(), expected), f"{axis.spherical()} != {expected}"


@pytest.mark.parametrize("param, expr", [
    [[0, 0], 'z'],
    [[np.pi, 0], '-z'],
    [[0, 2 * np.pi], 'z'],
    [[0, -2 * np.pi], 'z'],
    [[-np.pi / 2, 0], '-x'],
    [[-np.pi / 2, np.pi / 2], '-y'],
    [[-np.pi / 3, np.pi / 4], '(π/3, -3π/4)'],
    [[1e-5, 0, 0], 'x'],
    [[1, 0, 0], 'x'],
    [[0, 1, 0], 'y'],
    [[0, 0, 1], 'z'],
    [[0, .5, 0], 'y'],
    [[0, 0, -1], '-z'],
    [[0, 0, -.5], '-z'],
    [[-1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)], f'(0.955078125, 3π/4)']
])
def test_axis_repr(param, expr):
    axis = RAxis(np.array(param))
    assert repr(axis) == expr, f'Repr of Sphereical {param} != {repr(expr)}'
