import numpy as np

from quompiler.construct.cgate import CtrlGate
from quompiler.construct.types import UnivGate, QType
from quompiler.utils.group_su2 import euler_params


def euler_decompose(gate: CtrlGate) -> list[CtrlGate]:
    """
    Given a controlled unitary matrix, decompose it into Euler rotations and phase shift, e.g.,
    U = a @ Rz(b) @ Ry(c) @ Rz(d)
    A = Rz(b) @ Ry(c/2)
    B = Ry(-c/2) @ Ry(-(d+d)/2)
    C = Rz((d-b)/2)
    :param gate: controlled unitary matrix as input.
    :return: a list of Euler gates that's equivalent to the controlled unitary matrix.
    """
    assert gate.issinglet()
    a, b, c, d = euler_params(gate.matrix())
    A = UnivGate.Z.rotation(b) @ UnivGate.Y.rotation(c / 2)
    B = UnivGate.Y.rotation(-c / 2) @ UnivGate.Z.rotation(-(d + b) / 2)
    C = UnivGate.Z.rotation((d - b) / 2)
    target = gate.target_qids()

    if len(gate.control_qids()) == 1:
        _, ctrl = sorted(gate.controls)
        p1, p2 = (1, a) if ctrl.base[0] == 1 else (a, 1)
        phase_gate = CtrlGate(np.array([[p1, 0], [0, p2]]), [QType.TARGET], gate.control_qids())
    else:
        phase_gate = CtrlGate(a * np.eye(2), gate.controls, gate.qspace)
    result = [phase_gate,
              CtrlGate(A, [QType.TARGET], target),
              CtrlGate(UnivGate.X, gate.controls, gate.qspace),
              CtrlGate(B, [QType.TARGET], target),
              CtrlGate(UnivGate.X, gate.controls, gate.qspace),
              CtrlGate(C, [QType.TARGET], target)]
    return result
