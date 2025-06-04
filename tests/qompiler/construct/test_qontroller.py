import random
from functools import reduce

import pytest

from quompiler.construct.qontroller import Qontroller, ctrl2core
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
    # print(bin(num))
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

        # execute
        indexes = controller.indexes(universe)

        # verify
        expected = ctrl2core(controls)
        assert indexes == expected


def test_qontroller_indexes_target_idler_combined():
    controls = [QType.TARGET, QType.CONTROL1, QType.IDLER, QType.TARGET, QType.IDLER, QType.CONTROL0]
    controller = Qontroller(controls)
    qtype = QType.IDLER | QType.TARGET

    # execute
    indexes = controller.indexes(qtype)

    # verify
    actual = ctrl2core(controls)
    assert len(indexes) == len(actual)
    # print()
    # print(actual)
    # assert indexes == controller.extended_indexes()
    result = [a - b for a, b in zip(indexes, actual)]
    # print(result)
    assert all(b - a == 16 for a, b in zip(indexes, actual))
