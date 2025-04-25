import pytest

from quompiler.construct.types import QType
from quompiler.construct.controller import Controller
from quompiler.utils.cgen import random_control2

import random

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
    new = controller.mask(num)
    assert new == 16


def test_controller_mask_ones():
    controls = [QType.TARGET, QType.CONTROL1, QType.IDLER, QType.TARGET, QType.IDLER, QType.CONTROL0]
    controller = Controller(controls)
    n = len(controls)
    num = (1 << n) - 1
    print(bin(num))
    new = controller.mask(num)
    assert new == num - 1


def test_controller_indexes():
    controls = [QType.TARGET, QType.CONTROL1, QType.IDLER, QType.TARGET, QType.IDLER, QType.CONTROL0]
    controller = Controller(controls)
    indexes = controller.indexes()
    n = len(controls)
    num = 1 << n
    expected = sorted(set(controller.mask(i) for i in range(num)))
    assert indexes == expected
