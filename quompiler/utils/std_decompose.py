from quompiler.construct.cgate import CtrlGate
from quompiler.construct.types import UnivGate, EmitType
from quompiler.utils.euler_decompose import euler_decompose
from quompiler.utils.solovay import sk_approx


def cliffordt_decompose(gate: UnivGate) -> list[UnivGate]:
    # TODO
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
        if g.is_std() and univset == EmitType.UNIV_GATE:
            result.append(g)
        elif g.is_std():
            cts = cliffordt_decompose(g.gate)
            result.extend(CtrlGate(g, gate.controls, gate.qspace) for g in cts)
        else:
            mat = g._unitary.matrix
            apprxs = sk_approx(mat, rtol=rtol, atol=atol)
            result.extend(CtrlGate(g, gate.controls, gate.qspace) for g in apprxs)
    return result
