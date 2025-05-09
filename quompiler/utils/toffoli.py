from quompiler.construct.qspace import QSpace
from quompiler.construct.std_gate import CtrlStdGate


def toffoli(qspace: QSpace) -> list[CtrlStdGate]:
    """
    This is an example for ctrl_decompose() function for the CNOT gate specifically.
    This is a constant function, meaning both its input and output are both constants.
    :param gate: CNOT gate with two-qubit control sequence as input.
    :return: a list of CtrlStdGate and/or StdGate objects.
    """
    pass
