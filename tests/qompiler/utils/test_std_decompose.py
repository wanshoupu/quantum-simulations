from functools import reduce

import numpy as np
import pytest

from quompiler.construct.cgate import CtrlGate
from quompiler.construct.types import UnivGate
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_control, random_CtrlGate, random_phase

formatter = MatrixFormatter(precision=2)


@pytest.mark.parametrize("univgate", list(UnivGate))
def test_cliffordt_decompose_invariance(univgate):
    from quompiler.utils.std_decompose import cliffordt_decompose
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


def test_std_decompose_std_to_clifford(mocker):
    # mocking
    mock_sk_approx = mocker.patch('quompiler.utils.solovay.sk_approx', return_value=[])
    from quompiler.utils.std_decompose import std_decompose

    ctrls = random_control(3, 1)
    cg = CtrlGate(UnivGate.Y, ctrls)
    # print(f'cg:\n{formatter.tostr(cg.inflate())}')
    # print(cg.controllers)

    # execute
    ctrlgates = std_decompose(cg)

    # verify
    assert mock_sk_approx.call_count == 2
    assert len(ctrlgates) == 4


def test_std_decompose_2CU(mocker):
    # mocking
    mock_sk_approx = mocker.patch('quompiler.utils.solovay.sk_approx', return_value=[])
    from quompiler.utils.std_decompose import std_decompose

    ctrls = random_control(3, 1)
    cg = random_CtrlGate(ctrls)
    # print(f'cg:\n{formatter.tostr(cg.inflate())}')
    # print(cg.controllers)

    # execute
    ctrlgates = std_decompose(cg)

    # verify
    assert mock_sk_approx.call_count == 3
    assert len(ctrlgates) == 3
