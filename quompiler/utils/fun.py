from fractions import Fraction
from typing import Optional

import numpy as np

MAX_BINARY_LENGTH = 1 << 10


def rational(num: float, cofactor: float = 1, denom: int = 8) -> Optional[Fraction]:
    """
    Convert an angle to a rational multiple of Pi within a relative tolerance of 1e-5 and absolute tolerance of 1e-8.

    :param num: The angle in radians.
    :param cofactor: The cofactor to use. defaults to 1.
    :param denom: The maximum denominator allowed in the rational approximation. Should be a small enough integer, e.g., 2,3,4,5...
    :return: A tuple (numerator, denominator) such that angle ≈ (numerator / denominator) * π,
             or None if no such rational is found within the tolerances.
    """
    # Try to approximate the angle as a fraction of pi
    fraction = Fraction(num / cofactor).limit_denominator(denom)
    # Check error
    approx = fraction.limit_denominator(denom) * cofactor
    if np.isclose(num, float(approx)):
        return fraction
    return None


def pi_binary(angle: float) -> float:
    """
    Round the angle in terms of a binary fraction of π, like `k / 2^n * π`, k and n are integers.
    :param angle: input angle in radians
    :return: angle / π rounded to the nearest binary fraction.
    """
    result = round(angle * MAX_BINARY_LENGTH / np.pi)
    return result / MAX_BINARY_LENGTH


def pi_repr(angle: float) -> str:
    """
    Attempt to round the angle in terms of a binary fraction of π, or integer.
    As a last result, round to a binary fraction of π.
    :param angle:
    :return:
    """
    fr = rational(angle, np.pi)
    if fr is None:
        bf = pi_binary(angle)
        return f'{bf}π'
    sign = '-' if fr < 0 else ''
    fr = abs(fr)
    numerator = '' if fr.numerator == 1 else repr(fr.numerator)
    denominator = '' if fr.denominator == 1 else f'/{fr.denominator}'
    return f'{sign}{numerator}π{denominator}'
