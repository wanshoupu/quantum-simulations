from itertools import product
from typing import Sequence, Tuple, Iterable

import numpy as np

from quompiler.construct.types import QType
from numpy.typing import NDArray


def binary(bits: Sequence[int]) -> int:
    """
    This function converts sequence of binary bits into the integer represented by the bits.
    :param bits: the sequence of bits such as [1,0,0,1,0]. leading zeros have effect.
    :return: int represented by bits as a binary bits of an integer.
    """
    return sum(bit << i for i, bit in enumerate(reversed(bits)))


class QSpace:
    """
    Helper class for managing the bases of Hilbert space of n-qubits.
    Functionalities:
     1. shuffle a square matrix based on the sorting order of qids.
     2. map a sequence of unique indexes
    """

    def __init__(self, qids: Sequence[int]):
        assert len(set(qids)) == len(qids)
        self.qids: list[int] = list(qids)
        self.length = len(qids)
        self.sorting = np.argsort(qids).tolist()

    def __repr__(self):
        return repr(self.qids)

    def __eq__(self, __value):
        if __value is None:
            return False
        if not isinstance(__value, QSpace):
            return False
        return self.qids == __value.qids

    def __getitem__(self, index: int):
        return self.qids[index]

    def map_all(self, indexes: Sequence[int]) -> list[int]:
        """
        Convenient method built on top of self.map
        :param indexes: a sequence of indexes
        :return: a list of mapped indexes
        """
        return [self.map(i) for i in indexes]

    def map(self, n: int) -> int:
        """
        Shuffle the bits of an input integer viewed as binary according to the sorting order of qids.
        For example, if qid = [10,0,7,1,8], sorting order = [4,0,2,1,3]
        For an integer 0b10101, bits = [1,0,1,0,1] in Little Endian.
        After reordering the bits, we get output integer 0b11100
        :param n: input integer
        :return: perform bit shuffles on n and return the resulting integer.
        """
        assert n < (1 << self.length)
        result = 0
        for i, s in enumerate(self.sorting):
            # Get bit from s and put it at position i
            bit = (n >> s) & 1
            result |= (bit << i)
        return result

    def is_sorted(self) -> bool:
        return all(self.qids[i - 1] < self.qids[i] for i in range(1, len(self.qids)))


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

    def core(self) -> list[int]:
        """
        Create the core indexes in the controlled matrix.
        It is defined as the sparce indexes occupied by the matrix for targets + idlers under the controls (both type 0 and type 1) restrictions.
        :return: the core indexes in the controlled matrix.
        """
        if self._inflated_indexes is None:
            bases = [q.base for q in self.controls]
            self._inflated_indexes = [binary(bits) for bits in product(*bases)]
        return self._inflated_indexes

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

    def yeast(self) -> list[NDArray]:
        return [np.eye(2) for q in self.controls if q == QType.IDLER]

    def factors(self) -> list[int]:
        targets = np.array(self.controls) == QType.TARGET
        cumsum = np.cumsum(targets[::-1])[::-1].astype(int)
        rfactors = list(map(lambda a: 1 << int(a), cumsum[np.array(self.controls) == QType.IDLER]))
        return rfactors


def core2control(bitlength: int, core: Sequence) -> Tuple[QType, ...]:
    """
    Generate the control sequence of a bundle of indexes given by core.
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
