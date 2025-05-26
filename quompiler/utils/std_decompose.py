from quompiler.construct.cgate import CtrlGate
from quompiler.construct.types import UnivGate


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
