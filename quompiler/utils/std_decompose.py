from quompiler.construct.cgate import CtrlGate
from quompiler.construct.types import UnivGate, EmitType
from quompiler.utils.euler_decompose import euler_decompose
from quompiler.utils.solovay import sk_approx


def cliffordt_decompose(gate: CtrlGate) -> list[CtrlGate]:
    if gate.gate == UnivGate.Y:
        return [
            CtrlGate(UnivGate.X, gate.controls, gate.qspace, 1j * gate.phase()),
            CtrlGate(UnivGate.S, gate.controls, gate.qspace, 1),
            CtrlGate(UnivGate.S, gate.controls, gate.qspace, 1),
        ]
    if gate.gate == UnivGate.Z:
        return [CtrlGate(UnivGate.S, gate.controls, gate.qspace, gate.phase()),
                CtrlGate(UnivGate.S, gate.controls, gate.qspace, 1)]
    return [gate]


def std_decompose(gate: CtrlGate, univset: EmitType = EmitType.CLIFFORD_T, rtol=1.e-5, atol=1.e-8) -> list[CtrlGate]:
    """
    Given a single-qubit unitary matrix, decompose it into CtrlGate with or without controls.
    :param gate: CtrlGate as input.
    :param univset: The set of universal gates to be used for the decomposition. Default CLIFFORD + T.
    :param rtol: optional, if provided, will be used as the relative tolerance parameter.
    :param atol: optional, if provided, will be used as the absolute tolerance parameter.
    :return: a list of CtrlGate objects.
    """
    assert univset in {EmitType.CLIFFORD_T, EmitType.UNIV_GATE}
    assert gate.issinglet()
    coms = euler_decompose(gate)
    result = []
    for g in coms:
        if g.is_std():
            if univset == EmitType.UNIV_GATE or g.gate in UnivGate.cliffordt():
                result.append(g)
            else:
                cts = cliffordt_decompose(g)
                result.extend(cts)
        else:
            sk_coms = sk_approx(g.matrix(), rtol=rtol, atol=atol)
            result.extend(CtrlGate(g, gate.controls, gate.qspace) for g in sk_coms)
    return result
