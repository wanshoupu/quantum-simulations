"""
Decompose a d x d unitary matrix into product of two-level unitary matrices.
"""
import functools
from typing import List

import numpy as np
from numpy.typing import NDArray


def mat2l_decompose(m: np.ndarray) -> List[np.ndarray]:
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
            c = np.eye(*s).astype(np.complexfloating)
            if np.isclose(m[i, n], 0):
                # check if c will end up with identity
                if np.isclose(m[n, n], 1):
                    continue
                c[n, n] = np.conj(m[n, n])
            else:
                den = np.sqrt(np.conj(m[n, n]) * m[n, n] + np.conj(m[i, n]) * m[i, n])
                c[n, n] = np.conj(m[n, n]) / den
                c[i, n] = m[i, n] / den
                c[n, i] = np.conjugate(m[i, n] / den)
                c[i, i] = -m[n, n] / den
            result.append(np.conj(c).T)
            m = c @ m
    if not np.all(np.isclose(m, np.eye(*s))):
        result.append(m)
    return result


def validm2l(m: np.ndarray):
    """
    Validate if m is a 2-level unitary matrix.
    :param m: input matrix.
    :return: bool True if m is a 2-level unitary matrix; otherwise False.
    """
    n, k = m.shape
    if n != k:
        return False

    indexes = [i for i in range(n) if not np.isclose(m[i, i], 1)]
    if len(indexes) > 2:
        return False

    indx = [(i, j) for i, j in np.ndindex(m.shape) if i != j and not np.isclose(m[i, j], 0j)]
    if len(indx) > 2:
        return False
    return True


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
        assert np.all(np.isclose(recovered, m)), f'original\n{m}\n, recovered\n{recovered}'


    def _test_mat2l_2x2_noop():
        m = random_matrix_2l(2, 0, 1)
        print(formatter.tostr(m))
        tlms = mat2l_decompose(m)
        print(f'decompose =')
        for x in tlms:
            print(formatter.tostr(x), ',')
        recovered = functools.reduce(lambda a, b: a @ b, tlms)
        assert np.all(np.isclose(recovered, m)), f'original\n{formatter.tostr(m)}\n, recovered\n{formatter.tostr(recovered)}'


    def _test_mat2l_noop():
        m = random_matrix_2l(3, 0, 1)
        print(formatter.tostr(m))
        tlms = mat2l_decompose(m)
        print(f'decompose =')
        for x in tlms:
            print(formatter.tostr(x), ',')
        recovered = functools.reduce(lambda a, b: a @ b, tlms)
        assert np.all(np.isclose(recovered, m)), f'original\n{formatter.tostr(m)}\n, recovered\n{formatter.tostr(recovered)}'


    def _test_mat2l_3x3():
        m = random_unitary(3)
        print(f'original =')
        print(formatter.tostr(m))
        tlms = mat2l_decompose(m)
        print(f'decompose =')
        for x in tlms:
            print(formatter.tostr(x), ',')
        recovered = functools.reduce(lambda a, b: a @ b, tlms)
        assert np.all(np.isclose(recovered, m)), f'original\n{formatter.tostr(m)}\n, recovered\n{formatter.tostr(recovered)}'


    def _test_mat2l_random():
        for _ in range(10):
            n = random.randint(2, 8)
            m = random_unitary(n)
            print(f'original =')
            print(formatter.tostr(m))
            tlms = mat2l_decompose(m)
            print(f'decompose =')
            for x in tlms:
                print(formatter.tostr(x), ',\n')
            recovered = functools.reduce(lambda a, b: a @ b, tlms)
            assert np.all(np.isclose(recovered, m)), f'original\n{formatter.tostr(m)}\n, recovered\n{formatter.tostr(recovered)}'


    formatter = MatrixFormatter(precision=5)
    random.seed(3)

    _test_mat2l_cyclic()
    _test_mat2l_noop()
    _test_mat2l_2x2_noop()
    _test_mat2l_3x3()
    _test_mat2l_random()
