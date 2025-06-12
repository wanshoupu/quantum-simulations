import numpy as np
from pydantic.utils import defaultdict
from typing_extensions import override

from quompiler.construct.bytecode import Bytecode
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.qspace import Qubit
from quompiler.construct.types import OptLevel, EmitType
from quompiler.optimize.optimizer import Optimizer
from quompiler.utils.granularity import granularity


def cancel_append(prefix: dict[Qubit, list], node, window_size, required_emit):
    assert isinstance(node.data, CtrlGate)
    if np.allclose(node.data.matrix(), np.eye(1 << len(node.data.target_qids()))):
        return
    stacks = [prefix[q] for q in node.data.qspace]
    max_length = min(len(s) for s in stacks)
    window_size = max_length if window_size == 0 else min(max_length, window_size)
    window_size = compatible_length(stacks, window_size, node.data.qontrol())
    # add node
    for stack in stacks:
        stack.append(node)
    product = node.data
    # TODO find all the possible suffix optimizations and choose the best one
    for k in range(2, window_size + 2):
        product = stacks[0][-k].data @ product
        if np.allclose(np.array(product), np.eye(product.order())):
            # cancel out
            skip_top(stacks, k)
            break
        if required_emit <= granularity(product):
            # combine
            stacks[0][-k].data = product
            skip_top(stacks, k - 1)
            break
    # pop skipped nodes
    for stack in stacks:
        while stack and stack[-1].skip:
            stack.pop()


def compatible_length(stacks, window_size, qontrol):
    for k in range(1, window_size + 1):
        predecessor = {s[-k] if len(s) >= k else None for s in stacks}
        if len(predecessor) > 1 or (None in predecessor):
            return k - 1
        if predecessor.pop().data.qontrol() != qontrol:
            return k - 1
    return window_size


def skip_top(stacks, k):
    for i in range(1, 1 + k):
        stacks[0][-i].skip = True


class SlidingWindowOptimizer(Optimizer):
    """
    Sliding window optimizer performs optimization on adjacent moments within a sliding window of certain size.
    Example optimization performed:
    - Replace subsequence HXH with Z, HYH with -1 and Y, etc.
    - Replace two consecutive T by S
    - Replace two consecutive S by Z
    - Replace two consecutive X by I
    - Eliminate I
    """

    def __init__(self, window: int = 0, emit=EmitType.CLIFFORD_T):
        """
        Create a sliding window optimizer. it will try to combine consecutive (window + 1) gates on the same qspace with identical control sequences.
        :param window: window size. Window is >= 0. when window = 0, the window size is infinite.
        :param emit: the minimal required granularity. The combined gate is guaranteed to have a granularity >= it.
        """
        assert window >= 0
        self.window = window
        self.emit = emit

    @override
    def level(self) -> OptLevel:
        return OptLevel.O0

    @override
    def optimize(self, root: Bytecode) -> Bytecode:
        prefix = defaultdict(list)
        stack = [root]
        while stack:
            node = stack.pop()
            if node.is_leaf():
                cancel_append(prefix, node, self.window, self.emit)
            # Add children in reverse order so leftmost child is processed first
            stack.extend(reversed(node.children))
        return root
