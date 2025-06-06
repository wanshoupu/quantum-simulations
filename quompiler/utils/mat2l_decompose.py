"""
Decompose a d x d unitary matrix into product of two-level unitary matrices.
"""
from typing import List

import numpy as np

from quompiler.utils.mat_utils import coreindexes, validm2l
from quompiler.construct.unitary import UnitaryM


def mat2l_decompose(m: UnitaryM) -> List[UnitaryM]:
    """
    Decompose an NDArray with the optional dimension given by n
    :param m: UnitaryM
    :return: List of 2-level UnitaryM so that the product of them is equal to the original matrix.
    """
    result = []
    matrix = m.matrix
    s = matrix.shape
    for n in range(s[0] - 1):
        if validm2l(matrix):
            break
        if np.allclose(matrix[n + 1:, n], 0):
            continue
        for i in range(n + 1, s[0]):
            # this is weird! I have to use complex to assign complex to it.
            # check if c will end up with identity
            if np.isclose(matrix[i, n], 0) and np.isclose(matrix[n, n], 1):
                continue
            den = np.sqrt(np.conj(matrix[n, n]) * matrix[n, n] + np.conj(matrix[i, n]) * matrix[i, n])
            c = np.array([[matrix[n, n] / den, np.conj(matrix[i, n]) / den],
                          [matrix[i, n] / den, -np.conj(matrix[n, n]) / den]])
            m2l = UnitaryM(m.dimension, (m.core[n], m.core[i]), c)
            result.append(m2l)
            matrix = np.conj(m2l.inflate()).T @ matrix
    indxs = coreindexes(matrix)
    if not indxs:
        indxs = (0, 1)
    m2l = matrix[np.ix_(indxs, indxs)]
    if not np.allclose(m2l, np.eye(2)):
        result.append(UnitaryM(s[0], indxs, m2l))
    return result
