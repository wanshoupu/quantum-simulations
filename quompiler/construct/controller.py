from itertools import product
from typing import Sequence

from quompiler.construct.types import QType


def binary(bits: Sequence[int]) -> int:
    """
    This function converts sequence of binary bits into the integer represented by the bits.
    :param bits: the sequence of bits such as [1,0,0,1,0]. leading zeros have effect.
    :return: int represented by bits as a binary bits of an integer.
    """
    return sum(bit << i for i, bit in enumerate(reversed(bits)))


class Controller:
    def __init__(self, controls: Sequence[QType]):
        self._controls = controls
        self._length = len(controls)

        # control masks
        self._control_masks = dict()
        for i, c in enumerate(controls):
            if c == QType.CONTROL1:
                self._control_masks[self._length - 1 - i] = 1
            elif c == QType.CONTROL0:
                self._control_masks[self._length - 1 - i] = 0

        # cache fields
        self._inflated_indexes = None
        self._lookup = {}

    def map(self, index):
        """
        Given an index, map it to the controlled index
        :param index: an integer, presumably the extension index
        :return:
        """
        for i, b in self._control_masks.items():
            index &= ~(1 << i)
            if b:
                index |= 1 << i
        return index

    def core(self):
        """
        Create the core indexes in the controlled matrix.
        It is defined as the sparce indexes occupied by the matrix for targets + idlers under the controls (both type 0 and type 1) restrictions.
        :return: the core indexes in the controlled matrix.
        """
        if self._inflated_indexes is None:
            bases = [q.base for q in self._controls]
            self._inflated_indexes = [binary(bits) for bits in product(*bases)]
        return self._inflated_indexes

    def indexes(self, qtype: QType):
        """
        Create the subindexes spanned by certain 'QType'
        :param qtype:
        :return:
        """
        if qtype not in self._lookup:
            bases = [c.base if c in qtype else (0,) for i, c in enumerate(self._controls)]
            self._lookup[qtype] = [binary(bits) for bits in product(*bases)]
        return self._lookup[qtype]
