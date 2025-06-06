from fractions import Fraction

import numpy as np
import pytest

from quompiler.utils.fun import rational


@pytest.mark.parametrize("num", [
    1 / 2.0,
    3 / 7,
    5 / 3,
    -5,
    -11 / 5,
    1 / 3,
    1 / 4,
])
def test_pi_rational_nth(num):
    angle = np.pi * num
    fra = rational(angle)
    assert fra == Fraction.from_float(num)
