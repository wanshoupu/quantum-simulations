import random
from functools import reduce

import numpy as np

from quompiler.circuits.factory_manager import FactoryManager
from quompiler.construct.cgate import CtrlGate
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_CtrlGate, random_control
from quompiler.utils.std_decompose import ctrl_decompose

formatter = MatrixFormatter(precision=2)
factory = FactoryManager().create_factory()
device = factory.get_device()


def test_ctr_decompose_2CU():
    """
    controls = [C,C,T] + ancilla
    """
    ctrls = random_control(3, 1)
    cg = random_CtrlGate(ctrls)
    # print(f'cg:\n{formatter.tostr(cg.inflate())}')
    # print(cg.controllers)

    # execute
    ctrlgates = ctrl_decompose(cg, device=device, clength=1)

    # verify
    assert len(ctrlgates) == 3

    result = reduce(lambda x, y: x @ y, ctrlgates)
    actual = clean_up_ancilla(result)
    assert isinstance(actual, CtrlGate)
    # print(f'actual:\n{formatter.tostr(actual.inflate())}')
    expected = cg.sorted()
    # print(f'expected:\n{formatter.tostr(expected.inflate())}')
    assert np.allclose(actual.inflate(), expected.inflate()), f'actual != expected: \n{formatter.tostr(actual.inflate())},\n\n{formatter.tostr(expected.inflate())}'


def clean_up_ancilla(result):
    ancillas = [q for q in result.qspace if q.ancilla]
    for q in ancillas:
        result = result.project(q, np.array([1, 0]))
    result = result.sorted()
    return result


def test_ctr_decompose_clength_eq_2():
    k = random.randint(3, 5)
    t = random.randint(1, 3)
    ctrls = random_control(k, t)
    cu = random_CtrlGate(ctrls)
    # print(cu.controller)
    # print(f'cu:\n{formatter.tostr(cu.inflate())}')
    ctrlgates = ctrl_decompose(cu, device=device, clength=2)
    assert len(ctrlgates) == 7
    result = reduce(lambda x, y: x @ y, ctrlgates)
    actual = clean_up_ancilla(result)
    assert isinstance(actual, CtrlGate)
    # print(f'actual:\n{formatter.tostr(recovered.inflate())}')
    expected = cu.sorted()
    assert actual.qspace == expected.qspace
    # print(f'expected:\n{formatter.tostr(expected.inflate())}')
    assert np.allclose(actual.inflate(), expected.inflate()), f'actual != expected: \n{formatter.tostr(actual.inflate())},\n\n{formatter.tostr(expected.inflate())}'
