import random
from functools import reduce
from itertools import product

import pytest

from quompiler.construct.controller import Controller, binary
from quompiler.construct.types import QType
from quompiler.utils.cgen import random_control2

random.seed(3)


def test_controller_init():
    n = 5
    controls = random_control2(n)
    controller = Controller(controls)
    assert controller is not None


def test_controller_mask_zero():
    controls = [QType.TARGET, QType.CONTROL1, QType.IDLER, QType.TARGET, QType.IDLER, QType.CONTROL0]
    controller = Controller(controls)
    num = 0
    new = controller.map(num)
    assert new == 16


def test_controller_mask_ones():
    controls = [QType.TARGET, QType.CONTROL1, QType.IDLER, QType.TARGET, QType.IDLER, QType.CONTROL0]
    controller = Controller(controls)
    n = len(controls)
    num = (1 << n) - 1
    print(bin(num))
    new = controller.map(num)
    assert new == num - 1


#
# def test_controller_mask_random():
#     controls = [QType.TARGET, QType.CONTROL1, QType.IDLER, QType.TARGET, QType.IDLER, QType.CONTROL0]
#     controller = Controller(controls)
#     n = len(controls)
#     for _ in range(10):
#         num = random.randrange(1 << n)
#         # print(bin(num))
#         new = controller.mask(num)
#         assert new == num


def test_controller_core():
    controls = [QType.TARGET, QType.CONTROL1, QType.IDLER, QType.TARGET, QType.IDLER, QType.CONTROL0]
    controller = Controller(controls)
    indexes = controller.inflated_indexes()
    n = len(controls)
    num = 1 << n
    expected = sorted(set(controller.map(i) for i in range(num)))
    assert indexes == expected


@pytest.mark.parametrize("qtype,expected", [
    (QType.IDLER, [0, 2, 8, 10]),
    (QType.TARGET, [0, 4, 32, 36]),
    (QType.CONTROL1, [16]),
    (QType.CONTROL0, [0]),
])
def test_controller_indexes(qtype, expected):
    controls = [QType.TARGET, QType.CONTROL1, QType.IDLER, QType.TARGET, QType.IDLER, QType.CONTROL0]
    controller = Controller(controls)
    indexes = controller.indexes(qtype)
    assert indexes == expected


def test_controller_inflated_indexes_equivalence():
    for _ in range(10):
        n = random.randint(1, 5)
        controls = random_control2(n)
        controller = Controller(controls)
        universe = reduce(lambda a, b: a | b, QType)
        indexes = controller.indexes(universe)
        expected = controller.inflated_indexes()
        assert indexes == expected


def test_controller_indexes_target_idler_combined():
    controls = [QType.TARGET, QType.CONTROL1, QType.IDLER, QType.TARGET, QType.IDLER, QType.CONTROL0]
    controller = Controller(controls)
    qtype = QType.IDLER | QType.TARGET
    indexes = controller.indexes(qtype)
    assert len(indexes) == len(controller.inflated_indexes())
    # assert indexes == controller.extended_indexes()
    result = [a - b for a, b in zip(indexes, controller.inflated_indexes())]
    print(result)
    assert all(b - a == 16 for a, b in zip(indexes, controller.inflated_indexes()))
