from typing import Sequence

from quompiler.construct.types import QType
import random


def random_control2(n) -> Sequence[QType]:
    """
    Generate a random control sequence with total n qubits, k target qubits, (n-k) control qubits
    :param n: positive integer
    :return: Control sequence
    """
    result = [random.choice(list(QType)) for _ in range(n)]
    return tuple(result)
