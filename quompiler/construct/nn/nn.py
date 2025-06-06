from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors
from typing_extensions import override

from quompiler.construct.bytecode import Bytecode
from quompiler.construct.nn.abstract import QNN
from quompiler.construct.types import UnivGate
from quompiler.utils.su2fun import vec


class NNN(QNN):
    """
    NearestNeighbors with algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}
    """

    def __init__(self, seqs: list[tuple[NDArray, tuple[UnivGate]]]):
        self._seqs = seqs
        self._root: NearestNeighbors = None

    @override
    def lookup(self, mat: NDArray) -> tuple[Bytecode, float]:
        key = vec(mat)
        distances, indices = self._root.kneighbors([key], n_neighbors=1)
        error, index = distances[0][0], indices[0][0]
        approx_U, approx_seq = self._seqs[index]
        return Bytecode(approx_U, [Bytecode(g) for g in approx_seq]), error


class AutoNN(NNN):
    """
    NearestNeighbors with algorithm : 'auto'
    """

    @override
    def __init__(self, seqs: list[tuple[NDArray, tuple[UnivGate]]]):
        self._seqs = seqs
        self._root = NearestNeighbors(n_neighbors=1, algorithm='auto')
        self._root.fit(np.array([vec(u) for u, _ in self._seqs]))


class BruteNN(NNN):
    """
    NearestNeighbors with algorithm 'brute'
    """

    @override
    def __init__(self, seqs: list[tuple[NDArray, tuple[UnivGate]]]):
        self._seqs = seqs
        self._root = NearestNeighbors(n_neighbors=1, algorithm='brute')
        self._root.fit(np.array([vec(u) for u, _ in self._seqs]))


class BallTreeNN(NNN):
    """
    NearestNeighbors with algorithm 'ball_tree'
    """

    @override
    def __init__(self, seqs: list[tuple[NDArray, tuple[UnivGate]]]):
        self._seqs = seqs
        self._root = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        self._root.fit(np.array([vec(u) for u, _ in self._seqs]))
