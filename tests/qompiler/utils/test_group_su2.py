import numpy as np
import pytest

from quompiler.construct.types import UnivGate
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.group_su2 import rangle, gc_decompose
from quompiler.utils.mfun import herm
from quompiler.utils.mgen import random_unitary, random_su2

formatter = MatrixFormatter(precision=2)


@pytest.mark.parametrize("gate,expected", [
    [UnivGate.I, 0],
    [UnivGate.X, np.pi],
    [UnivGate.Y, np.pi],
    [UnivGate.Z, np.pi],
    [UnivGate.H, np.pi],
    [UnivGate.S, np.pi / 2],
    [UnivGate.T, np.pi / 4],
    [UnivGate.SD, np.pi / 2],
    [UnivGate.TD, np.pi / 4],
])
def test_rangle_std_gates(gate, expected):
    actual = rangle(gate)
    assert np.isclose(actual, expected), f'{actual} != {expected}'


def test_rangle_diff_by_2pi():
    gate = random_su2()
    actual = rangle(gate)
    expected = rangle(-gate)
    print(actual, expected)


def test_rangle_random_su2():
    gate = random_su2()
    actual = rangle(gate)
    print(actual)


def test_rangle_2pi():
    gate = -UnivGate.I.matrix
    actual = rangle(gate)
    assert np.isclose(actual, 2 * np.pi), f'{actual} != {np.pi}'


def test_gc_decompose():
    expected = random_su2()
    # print('expected')
    # print(formatter.tostr(expected))
    v, w = gc_decompose(expected)
    actual = -v @ w @ herm(v) @ herm(w)
    # print('actual')
    # print(formatter.tostr(actual))
    assert np.allclose(actual, expected)
