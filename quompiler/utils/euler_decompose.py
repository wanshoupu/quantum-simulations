import numpy as np
from numpy._typing import NDArray

from quompiler.construct.cgate import CtrlGate
from quompiler.construct.types import UnivGate, QType


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
    a, b, c, d = euler_param(gate.matrix())
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


def euler_param(u: NDArray) -> tuple[complex, float, float, float]:
    """
    Given a U(2) matrix, decompose it into Euler angles + an overall scalar factor.
    :param u: U(2) matrix as input
    :return: scalar factor + Euler angles (b, c, d), such that u = a * Rz(b) @ Ry(c) @ Rz(d)
    """
    assert u.shape == (2, 2) and np.allclose(u.conj().T @ u, np.eye(2)), "u must be unitary"
    det = np.linalg.det(u)
    c2 = u[0, 0] * u[1, 1] / det
    s2 = -u[1, 0] * u[0, 1] / det
    # assert np.isclose(1, c2 + s2)
    plus = c2 if np.isclose(c2, 0) else np.angle(u[1, 1] / u[0, 0])
    minus = s2 if np.isclose(s2, 0) else np.angle(-u[1, 0] / u[0, 1])

    b = (plus + minus) / 2  # angle before Y rotation
    d = (plus - minus) / 2  # angle after Y rotation

    # Determine rotation angle around Y (theta)
    x = np.real_if_close(c2 - s2)
    y = np.real_if_close(2 * u[1, 0] * u[1, 1] / det / np.exp(1j * b))
    # assert np.isclose(x.imag, 0) and np.isclose(y.imag, 0)
    c = np.arctan2(y, x)
    # Global phase
    a = (u[1, 1] / (np.cos(c / 2) * np.exp(.5j * (b + d)))) if c2 > s2 else (u[1, 0] / (np.sin(c / 2) * np.exp(.5j * (b - d))))
    return a, b, c, d  # Global phase (a), and Euler angles (b, c, d)
