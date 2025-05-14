from dataclasses import dataclass
from typing import Sequence, Dict, Union
from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass
class Qubit:
    """
    Represents a single qubit with a non-negative integer ID.
    """
    qid: int

    def __post_init__(self):
        assert self.qid >= 0, f'Qubit {self.qid} must be >= 0'

    def __repr__(self):
        return f'q{self.qid}'

    def __str__(self):
        return f'q{self.qid}'

    def __eq__(self, __value):
        if not isinstance(__value, Qubit):
            return NotImplemented(f'Comparison is undefined for type {type(__value)}')
        return self.qid == __value.qid

    def __hash__(self):
        return hash(self.qid)

    def __lt__(self, __value):
        if not isinstance(__value, Qubit):
            return NotImplemented(f'Comparison is undefined for type {type(__value)}')
        return self.qid < __value.qid

    def __le__(self, __value):
        if not isinstance(__value, Qubit):
            return NotImplemented(f'Comparison is undefined for type {type(__value)}')
        return self.qid < __value.qid or self.qid == __value.qid

    def to_dict(self) -> Dict:
        return asdict(self)


class Ancilla(Qubit):
    """
    Represents an ancilla qubit. It's strongly recommended to use distinct qid ranges for main Qubits and Ancilla qubits.
    For example, make Qubit in the range [0-100] and Ancilla in [101-200].
    """

    def __init__(self, qid):
        super().__init__(qid)

    def __repr__(self):
        return f'a{self.qid}'

    def __str__(self):
        return f'a{self.qid}'


class QSpace:
    """
    Helper class for managing the bases of Hilbert space of n-qubits.
    Functionalities:
     1. shuffle a square matrix based on the sorting order of qids.
     2. map a sequence of unique indexes
    """

    def __init__(self, qids: Sequence[Qubit]):
        assert len(set(qids)) == len(qids)
        self.qids: list[Qubit] = list(qids)
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

    def __iter__(self):
        return iter(self.qids)

    def __len__(self):
        return len(self.qids)

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
