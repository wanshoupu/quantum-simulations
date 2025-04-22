import random
import textwrap

import pytest
import sympy
from sympy import Matrix, symbols, kronecker_product as kron, pretty, pprint

from quompiler.construct.cmat import QubitClass
from sandbox.sym.sym_gen import square_m

random.seed(3)


def random_control2(n) -> tuple[QubitClass, ...]:
    """
    Generate a random control sequence with total n qubits, k target qubits, (n-k) control qubits
    :param n: positive integer
    :param k: 0< k <= n
    :return: Control sequence
    """
    mid = [q.id for q in QubitClass]
    result = [QubitClass.get(random.choice(mid)) for _ in range(n)]
    return tuple(result)


@pytest.mark.parametrize("control", [
    [QubitClass.TARGET, QubitClass.NONE, QubitClass.CONTROL],
    [QubitClass.TARGET, QubitClass.CONTROL, QubitClass.NONE],
    [QubitClass.NONE, QubitClass.TARGET, QubitClass.CONTROL],
    [QubitClass.CONTROL, QubitClass.TARGET, QubitClass.NONE],
    [QubitClass.CONTROL, QubitClass.NONE, QubitClass.TARGET],
    [QubitClass.NONE, QubitClass.CONTROL, QubitClass.TARGET],
])
def test_control2mat_single_target(control):
    print(control)
    A = square_m(8)
    print()
    pprint(A, num_columns=10000)
