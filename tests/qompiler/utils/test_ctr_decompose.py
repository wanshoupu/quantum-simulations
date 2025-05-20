import random
from functools import reduce

import numpy as np

from quompiler.circuits.factory_manager import FactoryManager
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.qspace import Qubit
from quompiler.construct.std_gate import CtrlStdGate
from quompiler.construct.types import QType, UnivGate
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_CtrlGate, random_control, random_unitary
from quompiler.utils.std_decompose import ctrl_decompose

formatter = MatrixFormatter(precision=2)


def test_ctr_decompose_2CU():
    """
    controls = [C,C,T] + ancilla
    """
    ctrls = random_control(3, 1)
    cg = random_CtrlGate(ctrls)
    print(f'cg:\n{formatter.tostr(cg.inflate())}')
    # print(cg.controllers)
    factory = FactoryManager().create_factory()
    device = factory.get_device()

    # execute
    ctrlgates = ctrl_decompose(cg, device=device, clength=1)

    # verify
    assert len(ctrlgates) == 3

    actual = clean_up_ancilla(ctrlgates)
    assert isinstance(actual, CtrlGate)
    # print(f'actual:\n{formatter.tostr(actual.inflate())}')
    expected = cg.sorted()
    # print(f'expected:\n{formatter.tostr(expected.inflate())}')
    assert np.allclose(actual.inflate(), expected.inflate()), f'actual != expected: \n{formatter.tostr(actual.inflate())},\n\n{formatter.tostr(expected.inflate())}'


def clean_up_ancilla(gates):
    result = reduce(lambda x, y: x @ y, gates)
    ancillas = [q for q in result.qspace if q.ancilla]
    for q in ancillas:
        result = result.project(q, np.array([1, 0]))
    result = result.sorted()
    return result


def test_ctr_decompose():
    k = random.randint(3, 8)
    t = random.randint(1, 3)
    ctrls = random_control(k, t)
    cu = random_CtrlGate(ctrls)
    # print(cu.controller)
    factory = FactoryManager().create_factory()
    device = factory.get_device()
    ctrlgates = ctrl_decompose(cu, device=device, clength=1)
    assert len(ctrlgates) == 13
    actual = clean_up_ancilla(ctrlgates)
    assert isinstance(actual, CtrlGate)
    # print(f'actual:\n{formatter.tostr(recovered.inflate())}')
    expected = cu.sorted()
    assert actual.qspace == expected.qspace  # TODO failing assert
    # print(f'expected:\n{formatter.tostr(expected.inflate())}')
    assert np.allclose(actual.inflate(), expected.inflate()), f'actual != expected: \n{formatter.tostr(actual.inflate())},\n\n{formatter.tostr(expected.inflate())}'
