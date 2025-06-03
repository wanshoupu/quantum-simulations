import numpy as np
from numpy.typing import NDArray

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


def cliffordt_seqs(length: int) -> list[tuple[NDArray, tuple[UnivGate]]]:
    """
    Grow the ε-bound tree rooted at `node` until the minimum distance between parent and child is less than `error`.
    We grow the subtree by gc_decompose the node into its commutators
    :param node: the root to begin with.
    """
    pairs = [(np.array(UnivGate.I), (UnivGate.I,))]  # start with identity
    cliffordt = UnivGate.cliffordt()
    cliffordt.remove(UnivGate.I)

    stack = [(np.eye(2), [])]
    while stack:
        mat, seq = stack.pop(0)
        if len(seq) == length:
            continue
        for c in cliffordt:
            if seq and seq[-1] == c:  # avoid consecutive repeat
                continue
            new_seq = seq + [c]
            pairs.append((mat @ c, tuple(new_seq)))
            stack.append((mat @ c, new_seq))
    return pairs
