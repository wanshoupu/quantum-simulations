import random
from typing import Sequence

import pytest
import sympy

from quompiler.construct.cmat import QubitClass
from quompiler.construct.controller import Controller
from quompiler.utils.cgen import random_control2
from sandbox.sym.contmat import CUnitary
from sandbox.sym.inter_product import mesh_product
from sandbox.sym.sym_gen import symmat
from sandbox.sym.symmat_format import mat_print

random.seed(3)


@pytest.mark.parametrize("controls", [
    [QubitClass.TARGET, QubitClass.IDLER, QubitClass.CONTROL0],
    [QubitClass.TARGET, QubitClass.CONTROL0, QubitClass.IDLER],
    [QubitClass.IDLER, QubitClass.TARGET, QubitClass.CONTROL0],
    [QubitClass.CONTROL0, QubitClass.TARGET, QubitClass.IDLER],
    [QubitClass.CONTROL0, QubitClass.IDLER, QubitClass.TARGET],
    [QubitClass.IDLER, QubitClass.CONTROL0, QubitClass.TARGET],
    [QubitClass.TARGET, QubitClass.IDLER, QubitClass.CONTROL1],
    [QubitClass.TARGET, QubitClass.CONTROL1, QubitClass.IDLER],
    [QubitClass.IDLER, QubitClass.TARGET, QubitClass.CONTROL1],
    [QubitClass.CONTROL1, QubitClass.TARGET, QubitClass.IDLER],
    [QubitClass.CONTROL1, QubitClass.IDLER, QubitClass.TARGET],
    [QubitClass.IDLER, QubitClass.CONTROL1, QubitClass.TARGET],
])
def test_control2mat_single_target(controls):
    # print()
    # print(controls)
    A = symmat(2)
    mat_print(A)

    cu = CUnitary(A, controls)
    actual = cu.inflate()

    expected = another_inflate(A, controls)
    assert actual == expected


def test_control2mat_zero_target():
    controls = [QubitClass.CONTROL1, QubitClass.CONTROL1]
    print()
    print(controls)

    A = symmat(1)
    # mat_print(A)
    cu = CUnitary(A, controls)
    result = cu.inflate()

    expected = sympy.eye(1 << len(controls))
    expected[-1, -1] = A[0, 0]
    expected2 = another_inflate(A, controls)
    assert result == expected == expected2


def test_control2mat_two_targets():
    controls = [QubitClass.TARGET, QubitClass.IDLER, QubitClass.TARGET, QubitClass.CONTROL1]
    A = symmat(1 << controls.count(QubitClass.TARGET))
    mat_print(A)
    for _ in range(10):
        random.shuffle(controls)
        print()
        print(controls)
        cu = CUnitary(A, controls)
        actual = cu.inflate()
        mat_print(actual)

        expected = another_inflate(A, controls)
        assert actual == expected


def another_inflate(A: sympy.Matrix, controls: Sequence[QubitClass]) -> sympy.Matrix:
    """
    This is another way to inflate matrix A with control sequences like TARGET, IDLER, CONTROL1, and CONTROL0.
    It calls mesh_product to inflate the IDLER bits. Then it calls Controller.indexes to inflate the control bits.
    :param A: the core matrix to be converted to controlled matrix.
    :param controls: the control sequences.
    :return: the inflated matrix.
    """
    n = len(controls)
    factors = [1 << controls[i + 1:].count(QubitClass.TARGET) for i, q in enumerate(controls) if q == QubitClass.IDLER]
    yeast = [sympy.eye(2) for _ in range(len(factors))]
    core = mesh_product(A, yeast, factors)

    result = sympy.eye(1 << n)
    controller = Controller(controls)
    for i, r in enumerate(controller.indexes()):
        for j, c in enumerate(controller.indexes()):
            result[r, c] = core[i, j]
    return result


def test_control2mat_random():
    for _ in range(10):
        n = random.randint(1, 5)
        controls = random_control2(n)
        print()
        print(controls)

        A = symmat(1 << controls.count(QubitClass.TARGET))
        # mat_print(A)
        cu = CUnitary(A, controls)
        actual = cu.inflate()
        mat_print(actual)
        expected = another_inflate(A, controls)
        assert actual == expected
