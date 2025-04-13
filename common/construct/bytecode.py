from typing import List, Optional
from numpy.typing import NDArray
from common.utils.format_matrix import MatrixFormatter

formatter = MatrixFormatter()


class Bytecode:
    def __init__(self, data, parent=None, children=None):
        self.parent: Optional[Bytecode] = parent
        self.data: NDArray = data
        self.children: List[Bytecode] = children or []

    def append(self, child: 'Bytecode'):
        self.children.append(child)

    def __repr__(self):
        if self.children:
            rows = ['Node data:', formatter.tostr(self.data), 'children:']
            for c in self.children:
                rows.append(repr(c))
            return '\n'.join(rows)
        return 'Leaf data:\n' + formatter.tostr(self.data)


class BytecodeIter:
    def __init__(self, bytecode: Bytecode):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass
