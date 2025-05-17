from typing import Sequence, Iterable

import numpy as np


class Permuter:
    """
    Represents a permuter that operates on permutations.

    This class provides two complementary methods: `permute` and `sort`.
    The `permute` method transforms a sorted sequence into a target permutation,
    mimicking the effect of shuffling. Conversely, the `sort` method restores
    a permuted sequence to its sorted order.
    """

    def __init__(self, seq: Sequence):
        """
        Construct a `Permuter` based on a target permutation represented as a permuted sequence.
        :param seq: permuted sequence from its original sorted state.
        """
        assert len(set(seq)) == len(seq)  # uniqueness
        self.length = len(seq)
        # The meaning of _sort is: [seq[i] for i in _sort] == sorted(seq)
        self._sort = np.argsort(seq).tolist()
        # The meaning of _perm is: given srt = sorted(seq), [srt[i] for i in _perm] == seq
        self._perm = np.argsort(self._sort).tolist()

    def __repr__(self):
        return repr(self._sort)

    def sort(self, seq: Sequence) -> Sequence:
        """
        'Sort' the input sequence according to the prescribed ordering of the constructor.
        :param seq:
        :return:
        """
        return [seq[i] for i in self._sort]

    def permute(self, seq: Sequence) -> Sequence:
        """
        The `permute` method transforms a sorted sequence into a target permutation,
        :param seq: sequence to be permuted.
        :return: permuted sequence.
        """
        return [seq[i] for i in self._perm]

    @classmethod
    def from_permute(cls, origin: Sequence, permute: Iterable) -> "Permuter":
        """
        Construct a `Permuter` based on the shuffling effect permuting the original to the target permutation.
        :param origin: original sequence.
        :param permute: permuted sequence based on the original sequence.
        :return:
        """
        assert len(set(origin)) == len(origin)
        assert sorted(origin) == sorted(permute)
        indexes = [origin.index(item) for item in permute]
        return Permuter(indexes)

    def bitsortall(self, indexes: Sequence[int]) -> list[int]:
        """
        Convenient method built on top of self.map
        :param indexes: a sequence of indexes
        :return: a list of mapped indexes
        """
        return [self.bitsort(i) for i in indexes]

    def bitsort(self, n: int) -> int:
        """
        Sort the bits of an input integer according to the sorting order prescribed by this Permuter.
        Big Endian is used when converting integer to binary sequence, e.g., 0b110010 -> [1,1,0,0,1,0]

        For example, seq = [10,0,7,1,8] produces sorting order = [1,3,2,4,0].
        According to this sorting order, an integer in binary form 0abcde, bits = [e,d,c,b,a] in Big Endian would be sorted to [d,b,c,a,e].
        The latter forms the integer binary, 0dbcae, again in Big Endian.
        :param n: input integer
        :return: perform bitsort on `n` and return the resulting integer.
        """
        return self._bitmap(n, self._sort)

    def bitpermute(self, n: int) -> int:
        """
        Permute the bits of an input integer according to the permutation order prescribed by this Permuter.
        Big Endian is used when converting integer to binary sequence, e.g., 0b110010 -> [1,1,0,0,1,0]

        For example, seq = [10,0,7,1,8] produces permutation order = [4,0,2,1,3].
        According to this permutation order, an integer in binary form 0abcde, bits = [e,d,c,b,a] in Big Endian would be permuted to [a,c,e,b,d]
        The latter forms the integer binary, 0acebd, again in Big Endian.
        :param n: input integer
        :return: perform bitpermute on `n` and return the resulting integer.
        """
        return self._bitmap(n, self._perm)

    def _bitmap(self, n: int, perm: Sequence[int]) -> int:
        """
        Map the bits of an input integer according to the permutation order given by `perm`.
        Big Endian is used when converting integer to binary sequence, e.g., 0b110010 -> [1,1,0,0,1,0]

        For example, if the permutation order = [4,0,2,1,3], an integer in binary form 0abcde, bits = [e,d,c,b,a] in Big Endian would be permuted to [a,c,e,b,d]
        The latter forms the integer binary, 0acebd, again in Big Endian.
        :param n: input integer
        :return: perform _bitmap on `n` and return the resulting integer.
        """
        length = len(perm)
        assert n < (1 << length), f'{n} exceeds what can be operated by bitpermute safely, namely {1 << len(perm)}'

        result = 0
        for i, s in enumerate(perm):
            # Get bit from position s
            bit = (n >> (length - 1 - s)) & 1
            # Put it at position i
            result |= bit << (length - 1 - i)
            print(bin(result))
        return result
