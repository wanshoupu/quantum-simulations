from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import numpy as np
from typing_extensions import override

from quompiler.construct.cgate import CtrlGate
from quompiler.construct.types import EmitType
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
        stacks = [self.prefix[q] for q in node.data.qspace]
        max_length = min(len(s) for s in stacks)
        window_size = max_length if self.window == 0 else min(max_length, self.window)
        k, product = self.compatible(stacks, window_size, node)
        # mark the tail to be skipped
        for i in range(1, k):
            stacks[0][-i].skip = True
        # pop skipped nodes
        for stack in stacks:
            while stack and stack[-1].skip:
                stack.pop()
        # add node
        if node.data != product:
            node.metadata['optimized'] = 'ConsolidateOperator'
        node.data = product
        for stack in stacks:
            stack.append(node)

    def compatible(self, stacks, window_size, node) -> tuple[int, Any]:
        assert window_size >= 0
        qontrol = node.data.qontrol()
        product = node.data
        for k in range(1, window_size + 1):
            predecessor = {s[-k] if len(s) >= k else None for s in stacks}
            if len(predecessor) > 1 or (None in predecessor):
                return k, product
            prior = predecessor.pop()
            if prior.data.qontrol() != qontrol:
                return k, product
            tentative = stacks[0][-k].data @ product
            if granularity(tentative) < self.emit:
                return k, product
            product = tentative
        return window_size + 1, product
