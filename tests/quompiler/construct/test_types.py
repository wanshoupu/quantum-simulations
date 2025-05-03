from functools import reduce
from operator import or_

import numpy as np
import pytest

from quompiler.construct.types import UnivGate, QType
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_unitary


def test_univ_gate_X():
    assert np.all(np.equal(UnivGate.X.mat[::-1], np.eye(2)))


def test_univ_gate_Y():
    mat = UnivGate.Y.mat
    assert np.array_equal(mat, UnivGate.Z.mat[[1, 0]] * 1j)


def test_univ_gate_get_none():
    m = random_unitary(2)
    a = UnivGate.get(m)
    assert a is None


def test_univ_gate_get_T():
    mat = np.sqrt(np.array([[1, 0], [0, 1j]]))
    gate = UnivGate.get(mat)
    assert gate == UnivGate.T


def test_univ_gate_get_Z():
    mat = UnivGate.H.mat @ UnivGate.X.mat @ UnivGate.H.mat
    gate = UnivGate.get(mat)
    assert gate == UnivGate.Z


def test_univ_gate_get_H():
    mat = (UnivGate.Z.mat + UnivGate.X.mat) / np.sqrt(2)
    gate = UnivGate.get(mat)
    assert gate == UnivGate.H


@pytest.mark.parametrize('name,expected', [
    ['I', UnivGate.I],
    ['X', UnivGate.X],
    ['Y', UnivGate.Y],
    ['Z', UnivGate.Z],
    ['H', UnivGate.H],
    ['S', UnivGate.S],
    ['T', UnivGate.T],
])
def test_univ_gate_get_by_name(name, expected):
    g = UnivGate[name]
    assert g == expected


def test_univ_gate_RY():
    from scipy.linalg import expm
    g = UnivGate.Z
    theta = np.pi * 2
    u = expm(-theta * 1j * g.mat / 2)
    formatter = MatrixFormatter(precision=2)
    # print(f'\n{g.name}=\n{formatter.tostr(g.mat)}')
    # print(f'\nexp(-{formatter.nformat(theta)}iY/2=\n{formatter.tostr(u)}')
    assert np.allclose(u, -np.eye(2))


def test_univ_gate_commutator():
    mat = UnivGate.Z.mat @ UnivGate.Y.mat @ UnivGate.Z.mat
    assert np.array_equal(mat, -UnivGate.Y.mat), f'mat unexpected {mat}'


def test_qtype_empty_state():
    # id zero is the empty enum, meaning no QType is present. It's the effective None QType
    empty_type = QType(0)
    assert all(t not in empty_type for t in QType)


@pytest.mark.parametrize("eid, qtype", [
    [1, QType.IDLER],
    [2, QType.TARGET],
    [4, QType.CONTROL0],
    [8, QType.CONTROL1],
])
def test_qtype_id(eid, qtype):
    singleton = QType(value=eid)
    assert singleton == qtype and qtype in singleton


@pytest.mark.parametrize("tid, qtypes", [
    [0b11, [QType.IDLER, QType.TARGET]],
    [0b101, [QType.IDLER, QType.CONTROL0]],
    [0b110, [QType.TARGET, QType.CONTROL0]],
    [0b111, [QType.CONTROL0, QType.IDLER, QType.TARGET]],
    [0b1100, [QType.CONTROL1, QType.CONTROL0]],
])
def test_qtype_combinations(tid, qtypes):
    combo = QType(value=tid)
    assert combo == reduce(or_, qtypes, QType(0))
    assert all(t in combo for t in qtypes)


def test_qtype_universe():
    universe = QType(value=0b1111)
    assert universe == reduce(or_, QType)
    assert all(t in universe for t in QType)


@pytest.mark.parametrize("tid", [0b11, 0b101, 0b110, 0b111, 0b1100, 0b1111])
def test_qtype_combo_in_combo(tid):
    universe = QType(value=0b1111)
    qtype = QType(tid)
    assert qtype in universe
    if qtype != universe:
        assert universe not in qtype


@pytest.mark.parametrize("name, qtype", [
    ["IDLER", QType.IDLER],
    ["TARGET", QType.TARGET],
    ["CONTROL0", QType.CONTROL0],
    ["CONTROL1", QType.CONTROL1],
    [None, QType.CONTROL1 | QType.IDLER],
])
def test_qtype_name(name, qtype):
    assert qtype.name == name


def test_qtype_in_operator():
    # Python's Flag and IntFlag enums override the __contains__ method
    combo = QType.CONTROL0 | QType.CONTROL1
    for qt in (QType.CONTROL0, QType.CONTROL1):
        assert qt in combo
    for qt in (QType.TARGET, QType.IDLER):
        assert qt not in combo


def test_qtype_equality():
    combo = QType.CONTROL0 | QType.CONTROL1
    for qt in (QType.CONTROL0, QType.CONTROL1):
        assert combo != qt
        assert (combo & qt) == qt
