from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic.utils import defaultdict
from typing_extensions import override

from quompiler.construct.bytecode import Bytecode
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.types import OptLevel, EmitType
from quompiler.optimize.optimizer import Optimizer
from quompiler.utils.granularity import granularity


class WindowOperator(ABC):
    def __init__(self, window: int = 0, emit=EmitType.CLIFFORD_T):
        assert window >= 0
        self.window = window
        self.emit = emit
        self.prefix = defaultdict(list)

    @abstractmethod
    def run(self, node):
        pass


class AnnihilateOperator(WindowOperator):
    def __init__(self, window: int = 0, emit=EmitType.CLIFFORD_T):
        super().__init__(window, emit)

    @override
    def run(self, node):
        assert isinstance(node.data, CtrlGate)
        if np.allclose(node.data.matrix(), np.eye(1 << len(node.data.target_qids()))):
            return
        stacks = [self.prefix[q] for q in node.data.qspace]
        max_length = min(len(s) for s in stacks)
        window_size = max_length if self.window == 0 else min(max_length, self.window)
        k = self.compatible_length(stacks, window_size, node.data.qontrol())
        # add node
        for stack in stacks:
            stack.append(node)
        product = node.data
        for k in range(2, k + 2):
            product = stacks[0][-k].data @ product
            if np.allclose(np.array(product), np.eye(product.order())):
                # cancel out the tail k nodes
                for i in range(1, 1 + k):
                    stacks[0][-i].skip = True
                # pop skipped nodes
                for stack in stacks:
                    while stack and stack[-1].skip:
                        stack.pop()
                return

    def compatible_length(self, stacks, window_size, qontrol):
        for k in range(1, window_size + 1):
            predecessor = {s[-k] if len(s) >= k else None for s in stacks}
            if len(predecessor) > 1 or (None in predecessor):
                return k - 1
            if predecessor.pop().data.qontrol() != qontrol:
                return k - 1
        return window_size


class ConsolidateOperator(WindowOperator):
    def __init__(self, window: int = 0, emit=EmitType.CLIFFORD_T):
        super().__init__(window, emit)

    @override
    def run(self, node):
        assert isinstance(node.data, CtrlGate)
        if np.allclose(node.data.matrix(), np.eye(1 << len(node.data.target_qids()))):
            return
        stacks = [self.prefix[q] for q in node.data.qspace]
        # add node
        for stack in stacks:
            stack.append(node)
        max_length = min(len(s) for s in stacks) - 1
        window_size = max_length if self.window == 0 else min(max_length, self.window)
        k, product = self.compatible(stacks, window_size, node)
        assert k > 0
        stacks[0][-k].data = product
        # mark the tail to be skipped
        for i in range(1, k):
            stacks[0][-i].skip = True
        # pop skipped nodes
        for stack in stacks:
            while stack and stack[-1].skip:
                stack.pop()

    def compatible(self, stacks, window_size, node) -> tuple[int, Any]:
        qontrol = node.data.qontrol()
        product = node.data
        for k in range(2, window_size + 2):
            predecessor = {s[-k] if len(s) >= k else None for s in stacks}
            product = stacks[0][-k].data @ product
            if len(predecessor) > 1 or (None in predecessor):
                return k - 1, product
            prior = predecessor.pop()
            if prior.data.qontrol() != qontrol:
                return k - 1, product
            if granularity(product) < self.emit:
                return k - 1, product

        return window_size + 1, product


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
        self.operators = [AnnihilateOperator(window, emit), ConsolidateOperator(window, emit)]

    @override
    def level(self) -> OptLevel:
        return OptLevel.O0

    @override
    def optimize(self, root: Bytecode) -> Bytecode:
        for op in self.operators:
            self.iterate(root, op)
        return root

    @staticmethod
    def iterate(root, operator):
        stack = [root]
        while stack:
            node = stack.pop()
            if node.is_leaf() and not node.skip:
                operator.run(node)
            # Add children in reverse order so leftmost child is processed first
            stack.extend(reversed(node.children))
