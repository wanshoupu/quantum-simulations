from dataclasses import dataclass, field
from typing import List, Union

import numpy as np
from numpy.typing import NDArray

from quompiler.construct.cgate import CtrlGate
from quompiler.construct.factor import FactoredM
from quompiler.construct.types import UnivGate
from quompiler.construct.unitary import UnitaryM
from quompiler.utils.mfun import herm


class Bytecode:
    def __init__(self, data: Union[NDArray, FactoredM, UnitaryM, CtrlGate, UnivGate], children: List['Bytecode'] = None, metadata: dict = None, skip=False):
        self.data = data
        self.children = children or []
        self.metadata = metadata or dict()
        self.skip = skip

    def __repr__(self):
        return f'Bytecode{{{self.skip},{repr(self.data)}}}'

    def add_child(self, child: 'Bytecode'):
        self.children.append(child)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def herm(self) -> 'Bytecode':
        """
        Create a Bytecode representing the Hermite of self
        :return: a new Bytecode representing the Hermite of self
        """
        if isinstance(self.data, np.ndarray):
            data = herm(self.data)
        elif isinstance(self.data, UnitaryM):
            data = self.data.herm()
        elif isinstance(self.data, CtrlGate):
            data = self.data.herm()
        elif isinstance(self.data, UnivGate):
            data = self.data.herm()
        else:
            raise NotImplementedError("Haven't implemented this yet.")
        return Bytecode(data, [c.herm() for c in self.children[::-1]])


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
