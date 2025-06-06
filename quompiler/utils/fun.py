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


def angle_repr(angle: float) -> str:
    fr = rational(angle, np.pi)
    if fr:
        if fr.numerator == 1:
            return f"π/{fr.denominator}"
        if fr.numerator == -1:
            return f"-π/{fr.denominator}"
        return f"{fr.numerator}π/{fr.denominator}"
    fr = rational(angle)
    if fr:
        return str(fr.numerator)

    return repr(round(angle * MAX_BINARY_LENGTH) / MAX_BINARY_LENGTH)
