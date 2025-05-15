from typing import Sequence, Iterable

import numpy as np


class Shuffler:
    def __init__(self, seq: Sequence):
        self.seq = seq
        self.length = len(seq)
        self.sorting = np.argsort(seq).tolist()

    def __repr__(self):
        return repr(self.sorting)

    def shuffle(self, seq: Sequence):
        return [seq[i] for i in self.sorting]

    @classmethod
    def from_permute(cls, origin: Sequence, permute: Iterable) -> "Shuffler":
        assert sorted(origin) == sorted(permute)
        indexes = [origin.index(item) for item in permute]
        return Shuffler(indexes)

    def map_all(self, indexes: Sequence[int]) -> list[int]:
        """
        Convenient method built on top of self.map
        :param indexes: a sequence of indexes
        :return: a list of mapped indexes
        """
        return [self.map(i) for i in indexes]

    def map(self, n: int) -> int:
        """
        Shuffle the bits of an input integer viewed as binary according to the sorting order given by self.sorting.
        For example, if self.seq = [10,0,7,1,8], sorting order = [1,3,2,4,0]
        For an integer in binary form 0abcde, bits = [e,d,c,b,a] in Little Endian.
        Reordering the bits, we get [e,a,c,b,d] from which for the integer in binary form 0dbcae
        :param n: input integer
        :return: perform bit shuffles on n and return the resulting integer.
        """
        assert n < (1 << self.length)
        result = 0
        for i, s in enumerate(self.sorting):
            # Get bit from position s
            bit = (n >> s) & 1
            # Put it at position i
            result |= (bit << i)
        return result
