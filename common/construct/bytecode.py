from dataclasses import dataclass, field
from typing import List, Optional

from common.construct.cmat import UnitaryM
from common.utils.format_matrix import MatrixFormatter

formatter = MatrixFormatter()


@dataclass
class Bytecode:
    data: UnitaryM
    children: List['Bytecode'] = field(default_factory=list)

    def append(self, child: 'Bytecode'):
        self.children.append(child)

    def __iter__(self):
        return BytecodeIter(self)


class BytecodeIter:
    def __init__(self, root: Bytecode):
        self.stack = [root]  # use a stack for pre-order

    def __iter__(self):
        return self

    def __next__(self) -> Bytecode:
        if not self.stack:
            raise StopIteration
        node = self.stack.pop()
        # Push children in reverse so left-most is processed first
        self.stack.extend(reversed(node.children))
        return node
