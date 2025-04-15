"""
Decompose a d x d unitary matrix into product of two-level unitary matrices.
"""
from typing import List

import numpy as np
from numpy.typing import NDArray

from common.construct.cmat import UnitaryM, coreindexes, validm2l


def mat2l_decompose(m: NDArray) -> List[UnitaryM]:
    result = []
    m = np.copy(m)
    s = m.shape
    for n in range(s[0] - 1):
        if validm2l(m):
            break
        if np.allclose(m[n + 1:, n], 0):
            continue
        for i in range(n + 1, s[0]):
            # this is weird! I have to use complex to assign complex to it.
            # check if c will end up with identity
            if np.isclose(m[i, n], 0) and np.isclose(m[n, n], 1):
                continue
            den = np.sqrt(np.conj(m[n, n]) * m[n, n] + np.conj(m[i, n]) * m[i, n])
            c = np.array([[m[n, n] / den, np.conj(m[i, n]) / den],
                          [m[i, n] / den, -np.conj(m[n, n]) / den]])
            m2l = UnitaryM(s[0], c, (n, i))
            result.append(m2l)
            m = np.conj(m2l.inflate()).T @ m
    ridxs, cidxs = coreindexes(m)
    m2l = m[np.ix_(ridxs, cidxs)]
    if not np.allclose(m2l, np.eye(2)):
        result.append(UnitaryM(s[0], m2l, ridxs, cidxs))
    return result
