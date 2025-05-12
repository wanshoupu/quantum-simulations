from typing import Sequence, Union

import numpy as np
from numpy.typing import NDArray

from quompiler.construct.cgate import CtrlGate
from quompiler.construct.qspace import Ancilla
from quompiler.construct.std_gate import CtrlStdGate
from quompiler.construct.types import UnivGate, QType
from quompiler.qompile.device import QDevice
from quompiler.utils.solovay import sk_approx


def toffoli(ctrs: Sequence[QType], qubits: Sequence[int]) -> list[CtrlStdGate]:
    return [CtrlStdGate(UnivGate.X, ctrs, qubits)]


def ctrl_decompose(gate: CtrlGate, device: QDevice, clength=1) -> list[CtrlGate]:
    """
    Given a single-qubit CtrlGate, decompose its control sequences into no more than 2.
    :param clength: maximum length of control sequence after the decomposition. 0<clength<=2. Default to 1
    :param gate: controlled unitary matrix as input
    :param device: a quantum device in charge of computational and/or ancilla qubit allocations.
    :return: a list of CtrlGate objects.
    """
    assert 1 <= clength <= 2

    # sort by controls
    gate = gate.sorted(np.argsort(list(gate.controller)))
    ctrl_seq = list(gate.controller)
    qspace = gate.qspace()
    assert len(ctrl_seq) == len(qspace)
    if len(ctrl_seq) <= clength:
        return [gate]
    an = len(ctrl_seq) - 1
    aspace = device.alloc_ancilla(an)
    prev_ctrl = [ctrl_seq[0]]
    prev_qubit = [qspace[0]]
    coms = []
    for i in range(1, len(ctrl_seq)):
        cnot_qubits = [prev_qubit, qspace[i], aspace[i - 1]]
        cnot_ctrl = [prev_ctrl, ctrl_seq[i], QType.CONTROL1]
        prev_ctrl = cnot_ctrl[-1]
        prev_qubit = cnot_qubits[-1]

        if clength == 1:
            coms.append(CtrlStdGate(UnivGate.X, cnot_ctrl, cnot_qubits))
        else:  # clength == 2
            toff = toffoli(cnot_ctrl, cnot_qubits)
            coms.extend(toff)
    core = CtrlGate(gate.unitary.matrix,[])
    return [gate]


def std_decompose(gate: Union[CtrlStdGate, CtrlGate], univset: Sequence[UnivGate], rtol=1.e-5, atol=1.e-8) -> list[CtrlStdGate]:
    """
    Given a single-qubit unitary matrix, decompose it into CtrlStdGate with or without controls.
    :param gate: controlled unitary matrix as input.
    :param univset: The set of universal gates to be used for the decomposition.
    :param rtol: optional, if provided, will be used as the relative tolerance parameter.
    :param atol: optional, if provided, will be used as the absolute tolerance parameter.
    :return: a list of CtrlStdGate objects.
    """
    seq = sk_approx(gate.inflate(), rtol=rtol, atol=atol)
    return [CtrlStdGate(g, gate.controller, gate.qspace) for g in seq]


def euler_decompose(cg: CtrlGate) -> list[CtrlGate]:
    """
    Given a controlled unitary matrix, decompose it into Euler rotations and phase shift, e.g.,
    U = a @ Rz(b) @ Ry(c) @ Rz(d)
    A = Rz(b) @ Ry(c/2)
    B = Ry(-c/2) @ Ry(-(d+d)/2)
    C = Rz((d-b)/2)
    :param cg: controlled unitary matrix as input.
    :return: a list of Euler gates that's equivalent to the controlled unitary matrix.
    """
    assert cg.issinglet()
    a, b, c, d = euler_param(cg.unitary.matrix)
    phase = a * np.eye(2)
    A = UnivGate.Z.rotation(b) @ UnivGate.Y.rotation(c / 2)
    B = UnivGate.Y.rotation(-c / 2) @ UnivGate.Z.rotation(-(d + b) / 2)
    C = UnivGate.Z.rotation((d - b) / 2)
    target = cg.target_qids()

    result = [CtrlGate(phase, cg.controller, cg.qspace),
              CtrlGate(A, [QType.TARGET], target),
              CtrlGate(UnivGate.X.matrix, cg.controller, cg.qspace),
              CtrlGate(B, [QType.TARGET], target),
              CtrlGate(UnivGate.X.matrix, cg.controller, cg.qspace),
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
