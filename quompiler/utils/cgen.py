from typing import Sequence

from quompiler.construct.cmat import QubitClass
import random


def random_control2(n) -> Sequence[QubitClass]:
    """
    Generate a random control sequence with total n qubits, k target qubits, (n-k) control qubits
    :param n: positive integer
    :param k: 0< k <= n
    :return: Control sequence
    """
    mid = [q.id for q in QubitClass]
    result = [QubitClass.get(random.choice(mid)) for _ in range(n)]
    return tuple(result)
