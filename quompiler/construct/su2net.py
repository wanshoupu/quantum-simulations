from abc import ABC
from logging import warning

from numpy._typing import NDArray

from quompiler.construct.bytecode import Bytecode
from quompiler.construct.nn.abstract import create_qnn
from quompiler.construct.types import SU2NetType


class SU2Net(ABC):

    def __init__(self, error, impl_type=SU2NetType.KDTreeNN):
        """
        A SU2 ε-net is a point cloud with distance between adjacent points no greater than `error`.
        This works for U2 just as well because the distance function is phase agnostic.
        It's organized in nary-tree similar to Geohash: the longer the sequence, the more precise it is.
        Implementation uses KDTree with `group_su2.vec` as the vectorization method.
        :param error: optional, if provided, will be used as the error tolerance parameter.
        """
        self.error = error
        self.length = int(1 / self.error) + 2
        self.constructed = False
        self._type = impl_type
        self._root = None

    def lookup(self, mat: NDArray) -> tuple[Bytecode, float]:
        """
        This is the nearest neighbor lookup function for the input matrix.
        :param mat: input 2x2 unitary matrix.
        :return: the nearest neighbor of the input matrix.
        """
        if not self.constructed:
            self.constructed = True
            self._root = create_qnn(self.length, self._type)

        # only assert these when debugging and skip when running in optimized mode (python -O ...)
        assert mat.shape == (2, 2), f'Mat must be a single-qubit operator: mat.shape = (2, 2)'
        # assert np.allclose(mat @ herm(mat), np.eye(2)), f'mat must be unitary.'
        # assert np.isclose(np.linalg.det(mat), 1), f'Mat must have unit determinant.'
        node, error = self._root.lookup(mat)
        if self.error < error:
            warning(f'Search for {mat} did not converge to within the error range: {self.error}.')
        return node, error
