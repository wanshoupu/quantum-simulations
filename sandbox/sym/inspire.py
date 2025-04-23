import random
from itertools import product
from typing import Sequence

import numpy as np
import pytest
import sympy
from sympy import pprint

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
    print()
    print(controls)

    A = symmat(2)
    mat_print(A)

    cu = CUnitary(A, controls)
    result = cu.inflate()
    # mat_print(result)


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
    assert result == expected


def test_control2mat_two_targets():
    controls = [QubitClass.TARGET, QubitClass.IDLER, QubitClass.TARGET, QubitClass.CONTROL1]
    A = symmat(1 << controls.count(QubitClass.TARGET))
    mat_print(A)
    for _ in range(10):
        random.shuffle(controls)
        print()
        print(controls)
        cu = CUnitary(A, controls)
        result = cu.inflate()

        mat_print(result)

        # hack
        mask = Controller(controls)
        uncontrolled_indexes = sorted(set(mask.mask(i) for i in range(result.shape[0])))
        actual = result.extract(uncontrolled_indexes, uncontrolled_indexes)
        control_index = len(controls) - 1 - controls.index(QubitClass.CONTROL1)
        none_index = len(controls) - 1 - controls.index(QubitClass.IDLER)
        if control_index < none_index:
            none_index -= 1
        factors = [1 << none_index]
        expected = mesh_product(A, [sympy.eye(2)], factors)
        assert actual == expected


def test_control2mat_random():
    for _ in range(10):
        n = random.randint(1, 5)
        controls = random_control2(n)
        print()
        print(controls)

        A = symmat(1 << controls.count(QubitClass.TARGET))
        # mat_print(A)
        cu = CUnitary(A, controls)
        result = cu.inflate()
        mat_print(result)
