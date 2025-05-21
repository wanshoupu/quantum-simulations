from typing import Sequence

import numpy as np

from quompiler.construct.cgate import CtrlGate
from quompiler.construct.qspace import Qubit
from quompiler.construct.types import QType, UnivGate

_CNOT = lambda q1, q2: CtrlGate(UnivGate.X, [QType.CONTROL1, QType.TARGET], [q1, q2])


def _toffoli(t: Qubit, c1: Qubit, c2: Qubit) -> list[CtrlGate]:
    result = []
    result.append(CtrlGate(UnivGate.H, [QType.TARGET], [t]))
    result.append(_CNOT(c2, t))

    result.append(CtrlGate(UnivGate.TD, [QType.TARGET], [t]))
    result.append(_CNOT(c1, t))

    result.append(CtrlGate(UnivGate.T, [QType.TARGET], [t]))
    result.append(_CNOT(c2, t))

    result.append(CtrlGate(UnivGate.TD, [QType.TARGET], [t]))
    result.append(_CNOT(c1, t))

    result.append(CtrlGate(UnivGate.TD, [QType.TARGET], [c2]))
    result.append(_CNOT(c1, c2))
    result.append(CtrlGate(UnivGate.TD, [QType.TARGET], [c2]))
    result.append(_CNOT(c1, c2))
    result.append(CtrlGate(UnivGate.T, [QType.TARGET], [c1]))
    result.append(CtrlGate(UnivGate.S, [QType.TARGET], [c2]))

    result.append(CtrlGate(UnivGate.T, [QType.TARGET], [t]))
    result.append(CtrlGate(UnivGate.H, [QType.TARGET], [t]))

    return result


def toffoli_decompose(ctrls: Sequence[QType], qspace: Sequence[Qubit]) -> list[CtrlGate]:
    """
    This is an example for ctrl_decompose() function for the CNOT gate specifically.
    This is a constant function, meaning both its input and output are both constants.
    :param ctrls: CNOT gate with two-qubit control sequence as input.
    :param qspace: the qspace these gates operate on.
    :return: a list of CtrlStdGate and/or StdGate objects.
    """
    assert len(ctrls) == len(qspace) == 3
    sorting = np.argsort(ctrls)
    qtcc = [qspace[i] for i in sorting]
    ctcc = [ctrls[i] for i in sorting]
    assert ctcc[0] == QType.TARGET
    assert all(c in QType.CONTROL0 | QType.CONTROL1 for c in ctcc[1:])

    result = _toffoli(*qtcc)

    # adjust for control activation value
    for i in range(1, 3):
        if ctcc[i] == QType.CONTROL0:
            result = [CtrlGate(UnivGate.X, [QType.TARGET], [qtcc[i]])] + result
            result.append(CtrlGate(UnivGate.X, [QType.TARGET], [qtcc[i]]))

    return result
