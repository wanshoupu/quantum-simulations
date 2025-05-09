import random
from functools import reduce

import numpy as np
import pytest

from quompiler.construct.qontroller import Qontroller
from quompiler.construct.qspace import QSpace
from quompiler.construct.types import QType
from quompiler.utils.mgen import random_control


def test_qontroller_init():
    n = 5
    controls = random_control(n)
    controller = Qontroller(controls)
    assert controller is not None


def test_qontroller_mask_zero():
    controls = [QType.TARGET, QType.CONTROL1, QType.IDLER, QType.TARGET, QType.IDLER, QType.CONTROL0]
    controller = Qontroller(controls)
    num = 0
    new = controller.mask(num)
    assert new == 16


def test_qontroller_mask_ones():
    controls = [QType.TARGET, QType.CONTROL1, QType.IDLER, QType.TARGET, QType.IDLER, QType.CONTROL0]
    controller = Qontroller(controls)
    n = len(controls)
    num = (1 << n) - 1
    print(bin(num))
    new = controller.mask(num)
    assert new == num - 1


def test_qontroller_mask_random():
    controls = [QType.TARGET, QType.CONTROL1, QType.IDLER, QType.TARGET, QType.IDLER, QType.CONTROL0]
    controller = Qontroller(controls)
    n = len(controls)
    c1 = n - 1 - controls.index(QType.CONTROL1)
    c0 = n - 1 - controls.index(QType.CONTROL0)
    for _ in range(10):
        num = random.randrange(1 << n)
        # print(bin(num))
        new = controller.mask(num)
        expected = (num | (1 << c1)) & ((1 << n) - 1 - (1 << c0))
        assert new == expected


def test_qontroller_core():
    controls = [QType.TARGET, QType.CONTROL1, QType.IDLER, QType.TARGET, QType.IDLER, QType.CONTROL0]
    controller = Qontroller(controls)
    indexes = controller.core()
    n = len(controls)
    num = 1 << n
    expected = sorted(set(controller.mask(i) for i in range(num)))
    assert indexes == expected


def test_qontroller_core_create():
    for _ in range(10):
        n = random.randint(1, 10)
        dim = 1 << n
        k = random.randint(2, min(dim, 100))
        core = (random.sample(range(dim), k))
        controller = Qontroller.create(n, core)
        controlled_core = controller.core()
        assert set(core) <= set(controlled_core)
        expected = sorted(set(controller.mask(i) for i in range(dim)))
        assert controlled_core == expected


@pytest.mark.parametrize("qtype,expected", [
    (QType.IDLER, [0, 2, 8, 10]),
    (QType.TARGET, [0, 4, 32, 36]),
    (QType.CONTROL1, [16]),
    (QType.CONTROL0, [0]),
])
def test_qontroller_indexes(qtype, expected):
    controls = [QType.TARGET, QType.CONTROL1, QType.IDLER, QType.TARGET, QType.IDLER, QType.CONTROL0]
    controller = Qontroller(controls)
    indexes = controller.indexes(qtype)
    assert indexes == expected


def test_qontroller_inflated_indexes_equivalence():
    for _ in range(10):
        n = random.randint(1, 5)
        controls = random_control(n)
        controller = Qontroller(controls)
        universe = reduce(lambda a, b: a | b, QType)
        indexes = controller.indexes(universe)
        expected = controller.core()
        assert indexes == expected


def test_qontroller_indexes_target_idler_combined():
    controls = [QType.TARGET, QType.CONTROL1, QType.IDLER, QType.TARGET, QType.IDLER, QType.CONTROL0]
    controller = Qontroller(controls)
    qtype = QType.IDLER | QType.TARGET
    indexes = controller.indexes(qtype)
    assert len(indexes) == len(controller.core())
    print()
    print(controller.core())
    # assert indexes == controller.extended_indexes()
    result = [a - b for a, b in zip(indexes, controller.core())]
    print(result)
    assert all(b - a == 16 for a, b in zip(indexes, controller.core()))


def test_qontroller_yeast_factor():
    controls = [QType.TARGET, QType.CONTROL1, QType.IDLER, QType.TARGET, QType.IDLER, QType.CONTROL0]
    controller = Qontroller(controls)
    yeast = controller.yeast()
    factors = controller.factors()
    assert len(yeast) == len(factors)
    assert all(np.array_equal(y, np.eye(2)) for y in yeast)
    filtered_controls = [q for q in controls if q == QType.TARGET or q == QType.IDLER]
    expected = [1 << filtered_controls[i:].count(QType.TARGET) for i, q in enumerate(filtered_controls) if q == QType.IDLER]
    assert factors == expected


def test_qontroller_yeast_factor_random():
    for _ in range(10):
        n = random.randint(1, 15)
        controls = random_control(n)
        controller = Qontroller(controls)
        yeast = controller.yeast()
        factors = controller.factors()
        assert len(yeast) == len(factors)
        assert all(np.array_equal(y, np.eye(2)) for y in yeast)
        filtered_controls = [q for q in controls if q == QType.TARGET or q == QType.IDLER]
        expected = [1 << filtered_controls[i:].count(QType.TARGET) for i, q in enumerate(filtered_controls) if q == QType.IDLER]
        assert factors == expected


def test_qspace_init_random():
    dim = 100
    for _ in range(10):
        qids = np.random.choice(dim, size=10, replace=False)
        qs = QSpace(qids)
        assert qs
        sorting = np.argsort(qids).tolist()
        # print(sorting)
        assert sorting == qs.sorting


@pytest.mark.parametrize("qids,n,expected", [
    [[10, 0, 7, 1, 8], 0b10101, 0b11100],
])
def test_qspace_map(qids, n, expected):
    qids = [10, 0, 7, 1, 8]
    qs = QSpace(qids)
    actual = qs.map(n)
    assert actual == expected


def test_qspace_map_random():
    dim = 100
    for _ in range(10):
        k = random.randint(1, 10)
        qids = np.random.choice(dim, size=k, replace=False)
        qs = QSpace(qids)
        indexes = list(range(1 << k))
        actual = [qs.map(i) for i in indexes]
        # print(f'mapped={actual}')
        assert sorted(actual) == indexes


@pytest.mark.parametrize("qids,expected", [
    [[10, 0, 7, 1, 8], False],
    [[0, 1, 7, 8, 10], True],
])
def test_qspace_is_sorted(qids, expected):
    qs = QSpace(qids)
    assert qs.is_sorted() == expected


def test_qspace_map_all_random():
    dim = 100
    for _ in range(10):
        k = random.randint(1, 10)
        qids = np.random.choice(dim, size=k, replace=False)
        qs = QSpace(qids)
        size = random.randint(1, 1 << k)
        indexes = np.random.choice(1 << k, size=size, replace=False)
        expected = [qs.map(i) for i in indexes]
        actual = qs.map_all(indexes)
        assert actual == expected
