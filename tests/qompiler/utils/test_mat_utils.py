import random

import numpy as np
import pytest

from quompiler.construct.qontroller import Qontroller, core2control, ctrl2core
from quompiler.construct.types import QType
from quompiler.utils.mat_utils import idindexes, coreindexes
from quompiler.utils.mgen import cyclic_matrix, random_indexes, random_control


def test_coreindexes():
    m = cyclic_matrix(8, 2)
    indxs = coreindexes(m)
    assert indxs == tuple(range(2, 8))


def test_core2control():
    for _ in range(10):
        core = [random.randint(10, 100) for _ in range(random.randint(2, 3))]
        # print(core)
        blength = max(i.bit_length() for i in core)
        gcb = core2control(blength, core)
        bitmatrix = np.array([list(bin(i)[2:].zfill(blength)) for i in core])
        # print(bitmatrix)
        expected = [(QType.CONTROL1 if int(bitmatrix[0, i]) else QType.CONTROL0) if len(set(bitmatrix[:, i])) == 1 else QType.TARGET for i in range(blength)]
        assert gcb == tuple(expected), f'gcb {gcb} != expected {expected}'


def test_core2control_empty_core():
    with pytest.raises(AssertionError):
        # core cannot be empty
        core2control(5, [])


def test_core2control_core_exceed_n():
    with pytest.raises(AssertionError):
        # core cannot be empty
        core2control(5, [1 << 5])


def test_core2control_big_endian():
    n = 3
    core = [2, 3]
    control = core2control(n, core)
    assert control == (QType.CONTROL0, QType.CONTROL1, QType.TARGET)


def test_core2control_single_index():
    for _ in range(10):
        print(f'Test round {_}...')
        n = random.randint(1, 5)
        dim = 1 << n
        core = random_indexes(dim, 1)
        assert len(core) == 1
        index = core[0]
        control = core2control(n, core)
        print(control)
        expected = tuple(QType.CONTROL1 if bool(index & 1 << i) else QType.CONTROL0 for i in range(n))[::-1]
        assert control == expected


@pytest.mark.parametrize('n,core,expected', [
    [3, list(range(3)), (QType.CONTROL0, QType.TARGET, QType.TARGET)],
    [4, list(range(3)), (QType.CONTROL0, QType.CONTROL0, QType.TARGET, QType.TARGET)],
    [4, list(range(3, 6)), (QType.CONTROL0, QType.TARGET, QType.TARGET, QType.TARGET)],
    [4, [2, 4], (QType.CONTROL0, QType.TARGET, QType.TARGET, QType.CONTROL0)],
    [4, [3, 5], (QType.CONTROL0, QType.TARGET, QType.TARGET, QType.CONTROL1)],
])
def test_core2control_unsaturated_core(n, core, expected):
    """
    this test is to verify that unsaturated core (m indexes where 2^(n-1) < m < 2^n for some n) creates the correct control sequence.
    """
    control = core2control(n, core)
    assert control == expected


def test_roundtrip_core_ctrl_core():
    for _ in range(10):
        n = random.randint(1, 1 << 10)
        k = random.randint(1, n)
        core = random_indexes(n, k)
        control = core2control(n, core)
        rectified = tuple(ctrl2core(control))
        assert set(core) <= set(rectified)
        assert all(rectified[i - 1] < rectified[i] for i in range(1, len(rectified)))


def test_roundtrip_ctrl_core_ctrl():
    for _ in range(10):
        n = random.randint(1, 5)
        t = random.randint(1, n)
        control = random_control(n, t)
        k = control.count(QType.TARGET)
        core = ctrl2core(control)
        assert len(core) == 1 << k
        actual = core2control(n, core)
        assert actual == control


def test_idindexes():
    m = cyclic_matrix(8, 2)
    indxs = idindexes(m)
    assert indxs == (0, 1)


def test_complementary_indexes():
    m = cyclic_matrix(8, 2)
    indxs = sorted(idindexes(m) + coreindexes(m))
    assert indxs == list(range(8))
