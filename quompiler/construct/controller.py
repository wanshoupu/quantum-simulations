from itertools import product
from typing import Sequence

from quompiler.construct.types import QType


class Controller:
    def __init__(self, controls: Sequence[QType]):
        self.controls = controls
        self.length = len(controls)
        self.masks = dict()
        for i, c in enumerate(controls):
            if c == QType.CONTROL1:
                self.masks[self.length - 1 - i] = 1
            elif c == QType.CONTROL0:
                self.masks[self.length - 1 - i] = 0
        self.core_indexes = None

    def mask(self, n):
        for i, b in self.masks.items():
            n &= ~(1 << i)
            if b:
                n |= 1 << i
        return n

    def indexes(self):
        """
        Create the core indexes in the controlled matrix.
        It is defined as the sparce indexes occupied by the matrix for targets + idlers under the control (both type 0 and type 1) restrictions.
        :return: the core indexes in the controlled matrix.
        """
        if self.core_indexes is None:
            tuples = [q.base for q in self.controls]
            self.core_indexes = [int(''.join(map(str, bits)), 2) for bits in product(*tuples)]
        return self.core_indexes
