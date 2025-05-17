from itertools import product
from typing import Sequence, Tuple, Union
from deprecated import deprecated

import numpy as np

from quompiler.construct.qspace import Qubit
from quompiler.construct.types import QType
from numpy.typing import NDArray


def binary(bits: Sequence[int]) -> int:
    """
    This function converts sequence of binary bits into the integer represented by the bits.
    Big Endian is used. The most significant digit appears first in a sequence.
    For example, the sequence [1,0,0,1,0] translate to integer is 0b1001 = 9.
    :param bits: the sequence of bits such as [1,0,0,1,0]. leading zeros have effect.
    :return: int represented by bits as a binary bits of an integer.
    """
    return sum(bit << i for i, bit in enumerate(reversed(bits)))


class Qontroller:
    def __init__(self, controls: Sequence[QType]):
        self.controls = controls
        self.length = len(controls)

        # control masks
        self._control_masks = dict()
        for i, c in enumerate(controls):
            if c == QType.CONTROL1:
                self._control_masks[self.length - 1 - i] = 1
            elif c == QType.CONTROL0:
                self._control_masks[self.length - 1 - i] = 0

        # cache fields
        self._inflated_indexes = None
        self._lookup = {}

    def __getitem__(self, index: int):
        return self.controls[index]

    def __iter__(self):
        return iter(self.controls)

    def __len__(self):
        return len(self.controls)

    def __repr__(self):
        return repr(self.controls)

    @classmethod
    def create(cls, n: int, core: Sequence[int]) -> 'Qontroller':
        """
        Create a Qontroller based on number of qubits and core indexes
        :param n: number of qubits
        :param core: a sequence of indexes
        :return: Qontroller
        """
        assert len(core) > 1
        controls = core2control(n, core)
        return Qontroller(controls)

    def mask(self, index):
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

    @deprecated
    def core(self) -> list[int]:  # TODO to be deprecated
        """
        Create the core indexes in the controlled matrix.
        It is defined as the sparce indexes occupied by the matrix for targets + idlers under the controls (both type 0 and type 1) restrictions.
        :return: the core indexes in the controlled matrix.
        """
        raise NotImplementedError('Deprecated, use ctrl2core() instead')

    def indexes(self, qtype: QType):
        """
        Create the subindexes spanned by certain 'QType'
        :param qtype:
        :return:
        """
        if qtype not in self._lookup:
            bases = [c.base if c in qtype else (0,) for i, c in enumerate(self.controls)]
            self._lookup[qtype] = [binary(bits) for bits in product(*bases)]
        return self._lookup[qtype]

    def is_sorted(self) -> bool:
        return all(self.controls[i - 1] < self.controls[i] for i in range(1, len(self.controls)))


def core2control(bitlength: int, core: Sequence[int]) -> Tuple[QType, ...]:
    """
    Generate the control sequence of a bundle of indexes given by core.

    Big Endian is used. The most significant digit corresponds to the first in the control sequence.
    The CONTROL0/CONTROL1 correspond to the shared bits by all the indexes in the core. The rest are QType.TARGET.
    The control sequence is formed by mapping the corresponding common bits in the core (0->CONTROL0, 1->CONTROL1).
    Big endian is used, namely, most significant bits on the left most end of the array.
    :param bitlength: total length of the control sequence
    :param core: the core indexes, i.e., the indexes of the target bits
    :return: Tuple[bool] corresponding to the control bits
    """
    assert core, f'Core cannot be empty'
    dim = 1 << bitlength
    assert max(core) < dim, f'Invalid core: some index in core are bigger than numbers representable by {dim} bits.'
    idiff = []
    for i in range(bitlength):
        mask = 1 << i
        if len({(a & mask) for a in core}) == 2:
            idiff.append(i)
    controls = [QType.CONTROL1 if core[0] & (1 << j) else QType.CONTROL0 for j in range(bitlength)]
    for i in idiff:
        controls[i] = QType.TARGET
    return tuple(controls[::-1])


def ctrl2core(controls: Sequence[QType]) -> list[int]:
    """
    Create the core indexes in the controlled matrix.
    Big Endian is used. The most significant digit comes from the first element in the control sequence.
    It is defined as the sparce indexes occupied by the matrix for targets + idlers under the controls (both type 0 and type 1) restrictions.
    :return: the core indexes in the controlled matrix.
    """
    bases = [q.base for q in controls]
    return [binary(bits) for bits in product(*bases)]


def rec_core(n: int, core: Sequence[int]) -> list[int]:
    """
    Rectify the core indexes in the given core by making them accessible by qubits controller.
    :param n: number of qubits, the dimension of the qspace.
    :param core: a sequence of integers, the indexes of the target bits.
    :return: rectified core.
    """
    ctrl = core2control(n, core)
    return ctrl2core(ctrl)
