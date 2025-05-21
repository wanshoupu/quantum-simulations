import random
from typing import Sequence

import pytest
import sympy

from quompiler.construct.qontroller import ctrl2core
from quompiler.construct.types import QType
from quompiler.utils.mgen import random_control
from sandbox.sym.ctrl_mat import CUnitary
from sandbox.sym.inter_product import mesh_product
from sandbox.sym.sym_gen import symmat
from sandbox.sym.symmat_format import mat_print


@pytest.mark.parametrize("controls", [
    [QType.TARGET, QType.IDLER, QType.CONTROL0],
    [QType.TARGET, QType.CONTROL0, QType.IDLER],
    [QType.IDLER, QType.TARGET, QType.CONTROL0],
    [QType.CONTROL0, QType.TARGET, QType.IDLER],
    [QType.CONTROL0, QType.IDLER, QType.TARGET],
    [QType.IDLER, QType.CONTROL0, QType.TARGET],
    [QType.TARGET, QType.IDLER, QType.CONTROL1],
    [QType.TARGET, QType.CONTROL1, QType.IDLER],
    [QType.IDLER, QType.TARGET, QType.CONTROL1],
    [QType.CONTROL1, QType.TARGET, QType.IDLER],
    [QType.CONTROL1, QType.IDLER, QType.TARGET],
    [QType.IDLER, QType.CONTROL1, QType.TARGET],
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
    controls = [QType.CONTROL1, QType.CONTROL1]
    # print()
    # print(controls)

    A = symmat(1)
    # mat_print(A)
    cu = CUnitary(A, controls)
    result = cu.inflate()

    expected = sympy.eye(1 << len(controls))
    expected[-1, -1] = A[0, 0]
    expected2 = another_inflate(A, controls)
    assert result == expected == expected2


def test_control2mat_two_targets():
    controls = [QType.TARGET, QType.IDLER, QType.TARGET, QType.CONTROL1]
    A = symmat(1 << controls.count(QType.TARGET))
    mat_print(A)
    for _ in range(10):
        random.shuffle(controls)
        # print()
        # print(controls)
        cu = CUnitary(A, controls)
        actual = cu.inflate()
        mat_print(actual)

        expected = another_inflate(A, controls)
        assert actual == expected


def another_inflate(A: sympy.Matrix, controls: Sequence[QType]) -> sympy.Matrix:
    """
    This is another way to inflate matrix A with control sequences like TARGET, IDLER, CONTROL1, and CONTROL0.
    It calls mesh_product to inflate the IDLER bits. Then it calls Controller.inflated_indexes to inflate the control bits.
    :param A: the core matrix to be converted to controlled matrix.
    :param controls: the control sequences.
    :return: the inflated matrix.
    """
    n = len(controls)
    factors = [1 << controls[i + 1:].count(QType.TARGET) for i, q in enumerate(controls) if q == QType.IDLER]
    yeast = [sympy.eye(2) for _ in range(len(factors))]
    core = mesh_product(A, yeast, factors)

    result = sympy.eye(1 << n)
    reced_core = ctrl2core(controls)
    for i, row in enumerate(reced_core):
        for j, col in enumerate(reced_core):
            result[row, col] = core[i, j]
    return result


def test_control2mat_random():
    for _ in range(10):
        n = random.randint(1, 5)
        controls = random_control(n)
        # print()
        # print(controls)

        A = symmat(1 << controls.count(QType.TARGET))
        # mat_print(A)
        cu = CUnitary(A, controls)
        actual = cu.inflate()
        mat_print(actual)
        expected = another_inflate(A, controls)
        assert actual == expected
