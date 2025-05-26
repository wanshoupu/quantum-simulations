from functools import reduce

import numpy as np
import pytest

from quompiler.construct.cgate import CtrlGate
from quompiler.construct.types import UnivGate
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_control, random_phase
from quompiler.utils.std_decompose import cliffordt_decompose

formatter = MatrixFormatter(precision=2)


@pytest.mark.parametrize("univgate", list(UnivGate))
def test_cliffordt_decompose_invariance(univgate):
    ctrls = random_control(3, 1)
    cg = CtrlGate(univgate, ctrls, phase=random_phase())

    # execute
    gates = cliffordt_decompose(cg)

    # verify
    assert all(g.qspace == cg.qspace for g in gates)
    assert all(g.controls == cg.controls for g in gates)

    actual = reduce(lambda x, y: x @ y, gates)
    # print(f'actual:\n{formatter.tostr(actual.inflate())}')
    expected = cg.inflate()
    # print(f'expected:\n{formatter.tostr(expected)}')
    assert np.allclose(actual.inflate(), expected), f'actual != expected: \n{formatter.tostr(actual.inflate())},\n\n{formatter.tostr(expected)}'


@pytest.mark.parametrize("univgate", list(UnivGate))
def test_cliffordt_subset(univgate):
    cg = CtrlGate(univgate, random_control(3, 1), phase=random_phase())

    # execute
    gates = cliffordt_decompose(cg)

    # verify
    assert all(g.gate in UnivGate.cliffordt() for g in gates)
