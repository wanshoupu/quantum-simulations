import random

import numpy as np
import pytest

from quompiler.construct.qontroller import Qontroller, core2control
from quompiler.construct.types import QType
from quompiler.utils.mat_utils import idindexes, coreindexes
from quompiler.utils.mgen import cyclic_matrix, random_indexes


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


def test_control2core_empty_core():
    with pytest.raises(AssertionError):
        # core cannot be empty
        core2control(5, [])


def test_control2core_big_endian():
    n = 3
    core = [2, 3]
    control = core2control(n, core)
    assert control == (QType.CONTROL0, QType.CONTROL1, QType.TARGET)


def test_control2core_single_index():
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
def test_control2core_unsaturated_core(n, core, expected):
    """
    this test is to verify that unsaturated core (m indexes where 2^(n-1) < m < 2^n for some n) creates the correct control sequence.
    """
    control = core2control(n, core)
    assert control == expected


def test_control2core_random():
    for _ in range(10):
        n = random.randint(1, 5)
        control = [random.choice(list(QType)) for _ in range(n)]
        k = control.count(QType.TARGET) + control.count(QType.IDLER)
        core = Qontroller(control).core()
        assert len(core) == 1 << k
        for i in range(len(control)):
            if control[i] == QType.TARGET or control[i] == QType.IDLER:
                control[i] = QType.TARGET
        expected = core2control(n, core)
        assert tuple(control) == expected


def test_idindexes():
    m = cyclic_matrix(8, 2)
    indxs = idindexes(m)
    assert indxs == (0, 1)


def test_complementary_indexes():
    m = cyclic_matrix(8, 2)
    indxs = sorted(idindexes(m) + coreindexes(m))
    assert indxs == list(range(8))
