from quompiler.circuits.qdevice import QDevice
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.types import QType, UnivGate
from quompiler.utils.toffoli import toffoli_decompose


def ctrl_decompose(gate: CtrlGate, device: QDevice, clength=1) -> list[CtrlGate]:
    """
    Given a single-qubit CtrlGate, decompose its control sequences into no more than 2.
    :param clength: maximum length of control sequence after the decomposition. 0<clength<=2. Default to 1
    :param gate: controlled unitary matrix as input
    :param device: a quantum device in charge of computational and/or ancilla qubit allocations.
    :return: a list of CtrlGate objects.
    """
    assert 1 <= clength <= 2

    ctrl_seq = list(gate.controls())
    qspace = gate.qspace
    # ctrl indexes
    cindexes = [i for i, c in enumerate(ctrl_seq) if c in QType.CONTROL0 | QType.CONTROL1]
    if len(cindexes) <= clength:
        # noop
        return [gate]
    # alloc ancilla
    aspace = device.alloc_ancilla(len(cindexes) - 1)
    coms = []
    for i, ancilla in enumerate(aspace):
        if i == 0:
            actrl = ctrl_seq[cindexes[0]], ctrl_seq[cindexes[1]], QType.TARGET
            aqubit = qspace[cindexes[0]], qspace[cindexes[1]], ancilla
        else:
            actrl = ctrl_seq[cindexes[i + 1]], QType.CONTROL1, QType.TARGET
            aqubit = qspace[cindexes[i + 1]], aspace[i - 1], ancilla

        if clength == 1:
            coms.extend(toffoli_decompose(actrl, aqubit))
        else:  # clength == 2
            coms.append(CtrlGate(UnivGate.X, actrl, aqubit))
    # target indexes
    tindexes = [i for i, c in enumerate(ctrl_seq) if c in QType.TARGET]
    core_ctrl = [QType.TARGET] * len(tindexes) + [QType.CONTROL1]
    core_qspace = [qspace[i] for i in tindexes] + [aspace[-1]]
    core = CtrlGate(gate.matrix(), core_ctrl, core_qspace)
    return coms + [core] + coms[::-1]
