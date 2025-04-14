"""
Decompose a d x d unitary matrix into product of two-level unitary matrices.
"""
import functools
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
        if np.all(np.isclose(m[n + 1:, n], np.zeros(s[0] - n - 1))):
            continue
        for i in range(n + 1, s[0]):
            # this is weird! I have to use complex64 to assign complex to it.
            # check if c will end up with identity
            if np.isclose(m[i, n], 0) and np.isclose(m[n, n], 1):
                continue
            den = np.sqrt(np.conj(m[n, n]) * m[n, n] + np.conj(m[i, n]) * m[i, n])
            c = np.eye(2).astype(np.complexfloating)
            c[0, 0] = np.conj(m[n, n]) / den
            c[1, 0] = m[i, n] / den
            c[0, 1] = np.conjugate(m[i, n] / den)
            c[1, 1] = -m[n, n] / den
            m2l = UnitaryM(s[0], np.conj(c).T, indexes=(n, i))
            result.append(m2l)
            m = m2l.inflate() @ m
    idxs = coreindexes(m)
    m2l = m[np.ix_(idxs, idxs)]
    if not np.all(np.isclose(m2l, np.eye(2))):
        result.append(UnitaryM(s[0], m2l, indexes=tuple(idxs)))
    return result


if __name__ == '__main__':
    from common.utils.mgen import cyclic_matrix, random_matrix_2l, random_unitary
    from common.utils.format_matrix import MatrixFormatter
    import random

    random.seed(3)


    def _test_mat2l_cyclic():
        m = cyclic_matrix(8, 1)
        print(formatter.tostr(m))
        tlms = mat2l_decompose(m)
        recovered = functools.reduce(lambda a, b: a @ b, tlms)
        assert np.all(np.isclose(recovered.inflate(), m)), f'original\n{m}\n, recovered\n{recovered}'


    def _test_mat2l_2x2_noop():
        m = random_matrix_2l(2, 0, 1)
        print(formatter.tostr(m))
        tlms = mat2l_decompose(m)
        print(f'decompose =')
        for x in tlms:
            print(formatter.tostr(x.inflate()), ',')
        recovered = functools.reduce(lambda a, b: a @ b, tlms)
        assert np.all(np.isclose(recovered.inflate(), m)), f'original\n{formatter.tostr(m)}\n, recovered\n{formatter.tostr(recovered.inflate())}'


    def _test_mat2l_3x3_2l():
        mp = UnitaryM(3, random_matrix_2l(2, 0, 1), indexes=(0, 1)).inflate()
        print(formatter.tostr(mp))
        tlms = mat2l_decompose(mp)
        print(f'decompose =')
        for x in tlms:
            print(formatter.tostr(x.inflate()), ',')
        recovered = functools.reduce(lambda a, b: a @ b, tlms)
        assert np.all(np.isclose(recovered.inflate(), mp)), f'original\n{formatter.tostr(mp)}\n, recovered\n{formatter.tostr(recovered.inflate())}'


    def _test_mat2l_noop():
        m = random_matrix_2l(3, 0, 1)
        print(formatter.tostr(m))
        tlms = mat2l_decompose(m)
        print(f'decompose =')
        for x in tlms:
            print(formatter.tostr(x.inflate()), ',')
        recovered = functools.reduce(lambda a, b: a @ b, tlms)
        assert np.all(np.isclose(recovered.inflate(), m)), f'original\n{formatter.tostr(m)}\n, recovered\n{formatter.tostr(recovered.inflate())}'


    def _test_mat2l_3x3():
        m = random_unitary(3)
        print(f'original =')
        print(formatter.tostr(m))
        tlms = mat2l_decompose(m)
        print(f'decompose =')
        for x in tlms:
            print(formatter.tostr(x.inflate()), ',')
        recovered = functools.reduce(lambda a, b: a @ b, tlms)
        assert np.all(np.isclose(recovered.inflate(), m)), f'original\n{formatter.tostr(m)}\n, recovered\n{formatter.tostr(recovered.inflate())}'


    def _test_mat2l_random():
        for _ in range(10):
            n = random.randint(2, 8)
            m = random_unitary(n)
            print(f'original =')
            print(formatter.tostr(m))
            tlms = mat2l_decompose(m)
            print(f'decompose =')
            for x in tlms:
                print(formatter.tostr(x.inflate()), ',\n')
            recovered = functools.reduce(lambda a, b: a @ b, tlms)
            assert np.all(np.isclose(recovered.inflate(), m)), f'original\n{formatter.tostr(m)}\n, recovered\n{formatter.tostr(recovered.inflate())}'


    formatter = MatrixFormatter(precision=5)
    random.seed(3)

    # _test_mat2l_2x2_noop()
    _test_mat2l_3x3_2l()
    # _test_mat2l_3x3()
    # _test_mat2l_cyclic()
    # _test_mat2l_noop()
    # _test_mat2l_random()
