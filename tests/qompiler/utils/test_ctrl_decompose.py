import random
from functools import reduce

import numpy as np
import pytest

from quompiler.circuits.factory_manager import FactoryManager
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.types import QType
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_CtrlGate, random_control
from quompiler.utils.ctrl_decompose import ctrl_decompose

formatter = MatrixFormatter(precision=2)
factory = FactoryManager().create_factory()
device = factory.get_device()


def test_ctrl_decompose_2CU():
    ctrls = random_control(3, 1)
    cg = random_CtrlGate(ctrls)
    # print(f'cg:\n{formatter.tostr(cg.inflate())}')
    # print(cg.controllers)

    # execute
    ctrlgates = ctrl_decompose(cg, device=device, clength=1)

    # verify
    ccount = sum(c in QType.CONTROL1 | QType.CONTROL0 for c in ctrls)
    gcount = 32 * (ccount - 1) + 4 * ctrls.count(QType.CONTROL0) + 1
    assert len(ctrlgates) == gcount

    actual = reduce(lambda x, y: x @ y, ctrlgates).dela()
    assert isinstance(actual, CtrlGate)
    # print(f'actual:\n{formatter.tostr(actual.inflate())}')
    expected = cg.sorted()
    # print(f'expected:\n{formatter.tostr(expected.inflate())}')
    assert np.allclose(actual.inflate(), expected.inflate()), f'actual != expected: \n{formatter.tostr(actual.inflate())},\n\n{formatter.tostr(expected.inflate())}'


@pytest.mark.parametrize("clength", [1, 2])
def test_ctrl_decompose_clength_2(clength):
    k = random.randint(3, 5)
    t = random.randint(1, 3)
    ctrls = random_control(k, t)
    cu = random_CtrlGate(ctrls)
    # print(cu.controller)
    # print(f'cu:\n{formatter.tostr(cu.inflate())}')
    ctrlgates = ctrl_decompose(cu, device=device, clength=clength)
    ccount = sum(c in QType.CONTROL1 | QType.CONTROL0 for c in ctrls)
    if clength == 1:
        gcount = 32 * (ccount - 1) + 4 * ctrls.count(QType.CONTROL0) + 1
    else:
        # clength == 2
        gcount = 2 * (ccount - 1) + 1
    assert len(ctrlgates) == gcount
    actual = reduce(lambda x, y: x @ y, ctrlgates).dela()
    assert isinstance(actual, CtrlGate)
    # print(f'actual:\n{formatter.tostr(recovered.inflate())}')
    expected = cu.sorted()
    assert actual.qspace == expected.qspace
    # print(f'expected:\n{formatter.tostr(expected.inflate())}')
    assert np.allclose(actual.inflate(), expected.inflate()), f'actual != expected: \n{formatter.tostr(actual.inflate())},\n\n{formatter.tostr(expected.inflate())}'
