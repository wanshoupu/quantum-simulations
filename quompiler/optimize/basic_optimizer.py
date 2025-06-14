from typing import Optional

from typing_extensions import override

from quompiler.construct.bytecode import Bytecode
from quompiler.construct.types import OptLevel, EmitType
from quompiler.optimize.optimizer import Optimizer
from quompiler.optimize.window import AnnihilateOperator, ConsolidateOperator


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
        prune(root)
        # ensure root is not skipped, which should be the case except for test purposes.
        root.skip = False
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


def prune(node: Bytecode) -> Optional[Bytecode]:
    """
    clean out the skipped nodes by discarding them.
    """
    if node.skip:
        return None
    if node.is_leaf():
        return node
    children = []
    for child in node.children:
        pruned = prune(child)
        if pruned:
            children.append(pruned)
    if not children:
        return None
    if len(children) == 1:
        return children[0]
    node.children = children
    return node
