from typing import Sequence

import numpy as np


class Qubit:
    def __init__(self, qid):
        self.qid = qid


class Ancilla(Qubit):
    def __init__(self, qid):
        super().__init__(qid)


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
