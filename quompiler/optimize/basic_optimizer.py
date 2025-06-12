from itertools import chain

import numpy as np
from pydantic.utils import defaultdict
from typing_extensions import override

from quompiler.construct.bytecode import Bytecode, BytecodeIter
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.qspace import Qubit
from quompiler.construct.types import OptLevel, UnivGate
from quompiler.optimize.optimizer import Optimizer


def cancel_append(prefix: dict[Qubit, list], node):
    assert isinstance(node.data, CtrlGate)
    stacks = [prefix[q] for q in node.data.qspace]
    predecessors = {s[-1] if s else None for s in stacks}
    if len(predecessors) > 1 or (None in predecessors):
        # None or more than one gates meaning the prior gates qspace are not aligned with this one
        append_node(node, prefix)
        return

    prior = next(iter(predecessors))
    product = np.array(prior.data @ node.data)
    if np.allclose(product, np.eye(product.shape[0])):
        node.skip = True
        prior.skip = True
        # pop prior
        for stack in stacks:
            stack.pop()
    else:
        append_node(node, prefix)


def append_node(node, prefix):
    for q in node.data.qspace:
        stack = prefix[q]
        stack.append(node)


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

    def __init__(self, window: int):
        self.window = window

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
                cancel_append(prefix, node)
            # Add children in reverse order so leftmost child is processed first
            stack.extend(reversed(node.children))
        return root
