from dataclasses import dataclass, field
from typing import List, Optional

from common.construct.cmat import UnitaryM
from common.utils.format_matrix import MatrixFormatter

formatter = MatrixFormatter()


@dataclass
class Bytecode:
    data: UnitaryM
    parent: Optional['Bytecode']=None
    children: List['Bytecode'] = field(default_factory=list)

    def append(self, child: 'Bytecode'):
        self.children.append(child)


class BytecodeIter:
    def __init__(self, bytecode: Bytecode):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass
