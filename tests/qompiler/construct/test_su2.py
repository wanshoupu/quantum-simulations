import random
from typing import Union

import numpy as np
import pytest

from quompiler.construct.su2gate import RGate, RAxis, _principal_axes
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mfun import allprop
from quompiler.utils.mgen import random_rgate
from quompiler.utils.su2fun import rot

formatter = MatrixFormatter(precision=2)


@pytest.mark.parametrize("axis, error", [
    ['h', "axis must be in {'x', 'y', 'z'}"],
    ['-t', "axis must be in {'x', 'y', 'z'}"],
    ['-s', "axis must be in {'x', 'y', 'z'}"],
    ['-x', "axis must be in {'x', 'y', 'z'}"],
    [np.array([1]), 'axis must be either a 2-vector or 3-vector'],
    [np.array([1, 2, 3, 4]), 'axis must be either a 2-vector or 3-vector'],
    [np.array([0, 0, 0]), "axis must not be zero."],
])
def test_axis_init_invalid(axis: Union[str, np.ndarray], error):
    with pytest.raises(ValueError) as e:
        RAxis(axis)
    assert str(e.value) == error


@pytest.mark.parametrize('principal', ['x', 'y', 'z'])
def test_axis_init_principal(principal):
    axis = RAxis(principal)
    assert repr(axis) == principal
    assert np.allclose(axis.nvec, _principal_axes[principal])


@pytest.mark.parametrize('spherical, expected', [
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


@pytest.mark.parametrize('spherical, expected', [
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
    assert np.allclose(axis.nvec, expected), f'{axis.nvec} != {expected}'


@pytest.mark.parametrize('nvec, expected', [
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
    assert np.allclose(axis.nvec, expected), f'{axis.nvec} != {expected}'


@pytest.mark.parametrize('nvec, expected', [
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


@pytest.mark.parametrize('param, expr', [
    [[0, 0], 'z'],
    [[0, 2 * np.pi], 'z'],
    [[0, -2 * np.pi], 'z'],
    [[-np.pi / 3, np.pi / 4], '(π/3, -3π/4)'],
    [[1e-5, 0, 0], 'x'],
    [[1, 0, 0], 'x'],
    [[0, 1, 0], 'y'],
    [[0, 0, 1], 'z'],
    [[0, .5, 0], 'y'],
    [[-1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)], f'(0.3037109375π, 3π/4)']
])
def test_axis_repr(param, expr):
    axis = RAxis(np.array(param))
    assert repr(axis) == expr, f'Repr of Spherical {param} != {repr(expr)}'


@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
def test_rgate_init_str(axis):
    angle = random.uniform(0, 2 * np.pi)
    gate = RGate(angle, axis)
    assert gate.axis.isprincipal()


@pytest.mark.parametrize('angle, axis, expected', [
    [np.pi, 'x', 'Rx(π)'],
    [-np.pi, [3, 4, 5], 'R(π/4, 0.294921875π)(-π)'],
])
def test_rgate_init_ndarray(angle: float, axis, expected):
    gate = RGate(angle, axis)
    assert repr(gate) == expected


@pytest.mark.parametrize('angle, axis, expected', [
    [np.pi, 'x', rot(np.array([1, 0, 0]), np.pi)],
    [-np.pi, [3, 4, 5], rot(np.array([3, 4, 5]), -np.pi)],
])
def test_rgate_verify_matrix(angle: float, axis, expected):
    gate = RGate(angle, axis)
    assert np.allclose(gate.matrix, expected), f'{gate.matrix} != {expected}'


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 10))
def test_rgate_axis_aligned_matmul(seed):
    random.seed(seed)
    np.random.seed(seed)

    theta = random.uniform(0, np.pi)
    phi = random.uniform(-np.pi, np.pi)
    angle1 = random.uniform(0, 2 * np.pi)
    angle2 = random.uniform(0, 2 * np.pi)
    a = RGate(angle1, [theta, phi])
    b = RGate(angle2, [theta, phi])

    c = a @ b
    actual_theta, actual_phi = c.axis.spherical()
    assert np.allclose(actual_theta, theta), f'{actual_theta} != {theta}'
    assert np.allclose(actual_phi, phi), f'{actual_phi} != {phi}'


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 10))
def test_rgate_matmul(seed):
    random.seed(seed)
    np.random.seed(seed)

    a = random_rgate()
    b = random_rgate()
    actual = a @ b
    expected = a.matrix @ b.matrix
    # actual and expected shall be equal other than a global phase
    assert allprop(actual.matrix, expected), f'\n{formatter.tostr(actual.matrix)} != \n{formatter.tostr(expected)}'
