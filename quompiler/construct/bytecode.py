from dataclasses import dataclass, field
from typing import List, Optional

from quompiler.construct.cmat import UnitaryM
from quompiler.utils.format_matrix import MatrixFormatter

formatter = MatrixFormatter()


@dataclass
class Bytecode:
    data: UnitaryM
    children: List['Bytecode'] = field(default_factory=list)

    def append(self, child: 'Bytecode'):
        self.children.append(child)


class BytecodeIter:
    """
    This iterator is a visitor design for class Bytecode.

    It visits the root node first and then each of the child nodes in order as they appear in field 'children'.
    """

    def __init__(self, root: Bytecode):
        self.stack = [root]  # use a stack for pre-order

    def __iter__(self):
        return self

    def __next__(self) -> Bytecode:
        if not self.stack:
            raise StopIteration
        node = self.stack.pop()
        # Push children in reverse so left-most is visited first
        self.stack.extend(reversed(node.children))
        return node


class ReverseBytecodeIter:
    """
    This iterator is a visitor design for class Bytecode.

    It visits the root node first and then each of the child nodes in order as they appear in field 'children'.
    """

    def __init__(self, root: Bytecode):
        self.stack = [root]  # use a stack for pre-order

    def __iter__(self):
        return self

    def __next__(self) -> Bytecode:
        if not self.stack:
            raise StopIteration
        node = self.stack.pop()
        # Push children so right-most is visited first
        self.stack.extend(node.children)
        return node
