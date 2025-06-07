import numpy as np
import pytest

from quompiler.utils.fun import rational, pi_repr


@pytest.mark.parametrize("num, cofactor, denom, expected", [
    [-5, 5, 1, -1],
    [1 / 2.0, 1, 2, 1 / 2],
    [3 / 7, 1, 7, 3 / 7],
    [5 / 3, 1, 3, 5 / 3],
    [-11 / 5, 1, 5, -11 / 5],
    [1 / 3, 1, 3, 1 / 3],
    [np.pi / 4, np.pi, 4, 1 / 4],
    [-5 * np.pi, 5 * np.pi, 1, -1],
    [np.pi / 2.0, np.pi, 2, 1 / 2],
    [3 * np.pi / 7, np.pi, 7, 3 / 7],
    [5 * np.pi / 3, np.pi, 3, 5 / 3],
    [-11 * np.pi / 5, np.pi, 5, -11 / 5],
    [np.pi / 3, np.pi, 3, 1 / 3],
    [np.e / 4, np.e, 4, 1 / 4],
    [-5 * np.e, 5 * np.e, 1, -1],
    [np.e / 2.0, np.e, 2, 1 / 2],
    [3 * np.e / 7, np.e, 7, 3 / 7],
    [5 * np.e / 3, np.e, 3, 5 / 3],
    [-11 * np.e / 5, np.e, 5, -11 / 5],
    [np.e / 3, np.e, 3, 1 / 3],
])
def test_rational(num: float, cofactor: float, denom: float, expected: float):
    actual = rational(num, cofactor, denom)
    assert actual is not None
    assert np.isclose(float(actual), expected)


@pytest.mark.parametrize("num, cofactor, denom, expected", [
    [np.pi, np.pi, 1, 1],
    [-5, 5, 1, -1],
    [1 / 2.0, 1, 2, 1 / 2],
    [3 / 7, 1, 7, 3 / 7],
    [5 / 3, 1, 3, 5 / 3],
    [-11 / 5, 1, 5, -11 / 5],
    [1 / 3, 1, 3, 1 / 3],
    [np.pi / 4, np.pi, 4, 1 / 4],
    [-5 * np.pi, 5 * np.pi, 1, -1],
    [np.pi / 2.0, np.pi, 2, 1 / 2],
    [3 * np.pi / 7, np.pi, 7, 3 / 7],
    [5 * np.pi / 3, np.pi, 3, 5 / 3],
    [-11 * np.pi / 5, np.pi, 5, -11 / 5],
    [np.pi / 3, np.pi, 3, 1 / 3],
    [np.e / 4, np.e, 4, 1 / 4],
    [-5 * np.e, 5 * np.e, 1, -1],
    [np.e / 2.0, np.e, 2, 1 / 2],
    [3 * np.e / 7, np.e, 7, 3 / 7],
    [5 * np.e / 3, np.e, 3, 5 / 3],
    [-11 * np.e / 5, np.e, 5, -11 / 5],
    [np.e / 3, np.e, 3, 1 / 3],
])
def test_rational(num: float, cofactor: float, denom: float, expected: float):
    rat = rational(num, cofactor, denom)
    assert rat is not None
    actual = float(rat)
    assert np.isclose(actual, expected), f'{actual} != {expected}'


@pytest.mark.parametrize("angle, expected", [
    [np.pi, "π"],
    [np.pi / 4, "π/4"],
    [-5 * np.pi, "-5π"],
    [np.pi / 2.0, "π/2"],
    [3 * np.pi / 7, "3π/7"],
    [5 * np.pi / 3, "5π/3"],
    [-11 * np.pi / 5, "-11π/5"],
    [np.pi / 3, "π/3"],
    [1, "0.318359375π"],
])
def test_pi_repr(angle: float, expected):
    actual = pi_repr(angle)
    assert actual == expected, f'{actual} != {expected}'
