from typing import Sequence

from quompiler.construct.cgate import CtrlGate
from quompiler.construct.qspace import Qubit
from quompiler.construct.types import QType, UnivGate


def toffoli(ctrs: Sequence[QType], qspace: Sequence[Qubit]) -> list[CtrlGate]:
    """
    This is an example for ctrl_decompose() function for the CNOT gate specifically.
    This is a constant function, meaning both its input and output are both constants.
    :param ctrs: CNOT gate with two-qubit control sequence as input.
    :param qspace: the qspace these gates operate on.
    :return: a list of CtrlStdGate and/or StdGate objects.
    """
    return [CtrlGate(UnivGate.X, ctrs, qspace)]
