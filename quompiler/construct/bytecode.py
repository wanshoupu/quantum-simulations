from dataclasses import dataclass, field
from typing import List, Union

from quompiler.construct.cgate import CtrlGate
from quompiler.construct.factor import FactoredM
from quompiler.construct.types import UnivGate
from quompiler.construct.unitary import UnitaryM
from quompiler.utils.format_matrix import MatrixFormatter
from numpy.typing import NDArray

formatter = MatrixFormatter()


@dataclass
class Bytecode:
    data: Union[NDArray, FactoredM, UnitaryM, CtrlGate,  UnivGate]
    children: List['Bytecode'] = field(default_factory=list)

    def append(self, child: 'Bytecode'):
        self.children.append(child)

    def is_leaf(self) -> bool:
        return len(self.children) == 0


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


class BytecodeRevIter:
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
