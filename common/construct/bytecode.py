from typing import List, Optional
import numpy as np

from common.utils.format_matrix import MatrixFormatter

formatter = MatrixFormatter()


class Bytecode:
    def __init__(self, data, parent=None, children=None):
        self.parent: Optional[Bytecode] = parent
        self.data: np.ndarray = data
        self.children: List[Bytecode] = children or []

    def append(self, child: 'Bytecode'):
        self.children.append(child)

    def __repr__(self):
        rows = ['Code:']
        if self.children:
            rows.append('children:')
            for c in self.children:
                rows.append(repr(c))
            return '\n'.join(rows)
        return 'leaf:\n' + formatter.tostr(self.data)


class BytecodeIter:
    def __init__(self, bytecode: Bytecode):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass
