from functools import reduce
from typing import Union, Sequence

import numpy as np
from numpy.typing import NDArray

from quompiler.construct.cmat import coreindexes
from quompiler.utils.inter_product import validate_factors, mesh_product, validm, mesh_factor


def ispow2(n):
    assert n >= 0
    return n & (n - 1) == 0


class FactorMat:
    """
    Represent a unitary matrix in factored format.
    """
    def __init__(self, matrices: Sequence[NDArray] = tuple(), factors: Sequence[int] = tuple()):
        """
        Instantiate a unitary matrix. The inflate method creates the extended matrix. See mesh_product for the requirements on the core, eyes, and factors.
        :param matrices: A list of integers (k1,k2,...) representing the identity matrices of corresponding dimension k1,k2, ..., for mesh_product.
        :param factors: A list of integers [f1, f2, ...] for mesh_product.
        """
        assert len(matrices) == len(factors) + 1, f'Lengths of matrices and factors must be equal but got matrices={len(matrices)} and factors={len(factors)}'
        validate_factors(factors)
        assert all(len(m.shape) == 2 and m.shape[0] == m.shape[1] for m in matrices), f'Matrices must be 2D square.'
        assert all(np.allclose(m @ m.conj().T, np.eye(m.shape[0])) for m in matrices), f'Matrices must be unitary.'
        self.matrices = list(matrices)
        self.factors = list(factors)

    def __matmul__(self, other: Union['FactorMat', np.ndarray]) -> Union['FactorMat', np.ndarray]:
        if isinstance(other, np.ndarray):
            return self.inflate() @ other
        if self.order() != other.order():
            raise ValueError('matmul: Input operands have dimension mismatch.')
        pairs = zip(self.matrices, other.matrices)
        compatible = all(selfy.shape == othery.shape for selfy, othery in pairs)
        if self.factors == other.factors and compatible:
            matrices = [selfy @ othery for selfy, othery in pairs]
            return FactorMat(matrices, self.factors)
        return FactorMat.deflate(self.inflate() @ other.inflate())

    def order(self):
        return reduce(lambda a, b: a * b, [y.shape[0] for y in self.matrices], 1)

    def inflate(self) -> NDArray:
        """
        Create a full-blown NDArray represented by FactorMat, namely mesh_product of the matrix, eyes, and factors.
        :return: The full-blown NDArray represented by FactorMat.
        """
        return mesh_product(self.matrices, self.factors)

    @classmethod
    def deflate(cls, m: NDArray) -> 'FactorMat':
        validm(m)
        indxs = coreindexes(m)
        if not indxs:
            indxs = (0, 1)
        matrix = m[np.ix_(indxs, indxs)]
        return FactorMat(*mesh_factor(matrix))

    def isid(self) -> bool:
        pairs = [(m, np.eye(m.shape[0])) for m in self.matrices]
        return all(np.allclose(m, e) for m, e in pairs)
