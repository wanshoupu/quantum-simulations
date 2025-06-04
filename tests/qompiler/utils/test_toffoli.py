import random
from functools import reduce

import numpy as np
import pytest

from quompiler.construct.cgate import CtrlGate
from quompiler.construct.qspace import Qubit
from quompiler.construct.types import QType, UnivGate
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.toffoli import toffoli_decompose, _toffoli

formatter = MatrixFormatter(precision=2)


def random_qubit(ancilla=None) -> Qubit:
    qid = random.randint(1, 10000)
    ancilla = ancilla or random.choice([False, True])
    return Qubit(qid, ancilla=ancilla)


def test_toffoli_cnot_equiv():
    qspace = [Qubit(1), Qubit(2), Qubit(3)]
    actual = reduce(lambda x, y: x @ y, _toffoli(*qspace))
    # print(f'actual:\n{formatter.tostr(actual.inflate())}')
    ctrls = [QType.TARGET, QType.CONTROL1, QType.CONTROL1]
    expected = CtrlGate(UnivGate.X, ctrls, qspace)
    assert actual.qspace == expected.qspace
    # print(f'expected:\n{formatter.tostr(expected.inflate())}')
    assert np.allclose(actual.inflate(), expected.inflate())


def test_toffoli_reverse_eq_forward():
    qspace = [Qubit(1), Qubit(2), Qubit(3)]
    tcoms = _toffoli(*qspace)
    forward = reduce(lambda x, y: x @ y, tcoms)
    reverse = reduce(lambda x, y: x @ y, tcoms[::-1])
    assert len(tcoms) == 16
    assert np.allclose(forward.inflate(), reverse.inflate())


@pytest.mark.parametrize("ctrls", [
    [QType.CONTROL0, QType.CONTROL0, QType.TARGET],
    [QType.CONTROL1, QType.CONTROL0, QType.TARGET],
    [QType.CONTROL1, QType.CONTROL1, QType.TARGET],
    [QType.CONTROL0, QType.CONTROL1, QType.TARGET],
])
def test_toffoli_decompose(ctrls):
    random.shuffle(ctrls)
    qspace = [random_qubit() for _ in range(len(ctrls))]
    cg = CtrlGate(UnivGate.X, ctrls, qspace)
    # print(f'cg:\n{formatter.tostr(cg.inflate())}')
    # print(cg.controllers)
    # execute
    toffoli_coms = toffoli_decompose(ctrls, qspace)

    # verify
    gcount = 16 + ctrls.count(QType.CONTROL0) * 2
    assert len(toffoli_coms) == gcount

    actual = reduce(lambda x, y: x @ y, toffoli_coms).sorted()
    # print(f'actual:\n{formatter.tostr(actual.inflate())}')
    expected = cg.sorted()
    # print(f'expected:\n{formatter.tostr(expected.inflate())}')
    assert actual.qspace == expected.qspace
    assert np.allclose(actual.inflate(), expected.inflate()), f'actual != expected: \n{formatter.tostr(actual.inflate())},\n\n{formatter.tostr(expected.inflate())}'
