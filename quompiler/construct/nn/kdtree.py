import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree
from typing_extensions import override

from quompiler.construct.bytecode import Bytecode
from quompiler.construct.nn.abstract import QNN
from quompiler.construct.types import UnivGate
from quompiler.utils.group_su2 import vec


class KDTreeNN(QNN):
    @override
    def __init__(self, seq: list[tuple[NDArray, tuple[UnivGate]]]):
        self._seqs = seq
        self._root = KDTree(np.array([vec(u) for u, _ in self._seqs]))

    @override
    def lookup(self, mat: NDArray) -> tuple[Bytecode, float]:
        key = vec(mat)
        distances, indices = self._root.query([key], k=1)
        error, index = distances[0], indices[0]
        approx_U, approx_seq = self._seqs[index]
        return Bytecode(approx_U, [Bytecode(g) for g in approx_seq]), error
