"""
Decompose a d x d unitary matrix into product of two-level unitary matrices.
"""
from typing import List

import numpy as np

from common.construct.cmat import UnitaryM, coreindexes, validm2l


def mat2l_decompose(m: UnitaryM) -> List[UnitaryM]:
    """
    Decompose an NDArray with the optional dimension given by n
    :param m: UnitaryM
    :return: List of 2-level UnitaryM so that the product of them is equal to the original matrix.
    """
    result = []
    mcopy = np.copy(m.matrix)
    s = mcopy.shape
    for n in range(s[0] - 1):
        if validm2l(mcopy):
            break
        if np.allclose(mcopy[n + 1:, n], 0):
            continue
        for i in range(n + 1, s[0]):
            # this is weird! I have to use complex to assign complex to it.
            # check if c will end up with identity
            if np.isclose(mcopy[i, n], 0) and np.isclose(mcopy[n, n], 1):
                continue
            den = np.sqrt(np.conj(mcopy[n, n]) * mcopy[n, n] + np.conj(mcopy[i, n]) * mcopy[i, n])
            c = np.array([[mcopy[n, n] / den, np.conj(mcopy[i, n]) / den],
                          [mcopy[i, n] / den, -np.conj(mcopy[n, n]) / den]])
            m2l = UnitaryM(m.dimension, c, (m.indexes[n], m.indexes[i]))
            result.append(m2l)
            mcopy = np.conj(m2l.inflate()).T @ mcopy
    indxs = coreindexes(mcopy)
    if not indxs:
        indxs = (0, 1)
    m2l = mcopy[np.ix_(indxs, indxs)]
    if not np.allclose(m2l, np.eye(2)):
        result.append(UnitaryM(s[0], m2l, indxs))
    return result
