import numpy as np
import random

from common.construct.cmat import UnitaryM


def random_unitary(n):
    """Generate a random n x n unitary matrix."""
    # Step 1: Generate a random complex matrix
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    # Step 2: Compute QR decomposition
    Q, R = np.linalg.qr(A)
    # Step 3: Ensure Q is unitary (QR decomposition sometimes returns non-unitary Q due to signs)
    # Adjust phases to make Q truly unitary
    D = np.diag(R) / np.abs(np.diag(R))
    Q = Q @ np.diag(D)
    return Q

def random_UnitaryM_2l(n, r1, r2) -> UnitaryM:
    rr = lambda: random.randint(0, 10)
    m = np.array([[complex(rr(), rr()), complex(rr(), rr())],
                  [complex(rr(), rr()), complex(rr(), rr())]])
    r1, r2 = min(r1, r2), max(r1, r2)
    return UnitaryM(n, m, indexes=(r1, r2))

def random_matrix_2l(n, r1, r2):
    rr = lambda: random.randint(0, 10)
    m = np.diag([1 + 0j] * n)
    r1, r2 = min(r1, r2), max(r1, r2)
    m[r1, r1] = complex(rr(), rr())
    m[r2, r1] = complex(rr(), rr())
    m[r1, r2] = complex(rr(), rr())
    m[r2, r2] = complex(rr(), rr())
    return m


def permeye(indexes):
    """
    Create a square identity matrix n x n, with the permuted indexes
    :param indexes: a permutation of indexes of list(range(len(indexes)))
    :return: the resultant matrix
    """
    return np.diag([1] * len(indexes))[indexes]

def xindexes(n, i, j):
    """
    Generate indexes list(range(n)) with the ith and jth swapped
    :param n: length of indexes
    :param i: ith index
    :param j: jth index
    :return: indexes list(range(n)) with the ith and jth swapped
    """
    indexes = list(range(n))
    indexes[i], indexes[j] = indexes[j], indexes[i]
    return indexes

def cyclic_matrix(n, i=0, j=None, c=1):
    """
    create a cyclic permuted matrix from identity
    :param n: dimension
    :param i: starting index of the cyclic permutation (inclusive). default 0
    :param j: ending index of the cyclic permutation (exclusive). default n
    :param c: shift cycles, default 1
    :return:
    """
    if j is None:
        j = n
    indexes = list(range(n))
    xs = indexes[:i] + np.roll(indexes[i:j], c).tolist() + indexes[j:]
    return permeye(xs)


if __name__ == '__main__':
    from common.utils.format_matrix import MatrixFormatter
    from common.utils.cnot_decompose import permeye

    # random.seed(3)
    formatter = MatrixFormatter()


    def _test_2l():
        m2l = random_matrix_2l(10, 1, 6)
        print(formatter.tostr(m2l))


    def _test_unitary():
        randu = random_unitary(2)
        print(formatter.tostr(randu))
        identity = randu.T @ np.conj(randu)
        assert np.all(np.isclose(identity, np.eye(*identity.shape))), print(formatter.tostr(identity))


    def _test_permeye():
        for _ in range(10):
            n = random.randint(10, 16)
            a = random.randrange(n)
            b = random.randrange(n)
            xs = xindexes(n, a, b)
            pi = permeye(xs)
            if a == b:
                assert pi[a, a] == 1 == pi[b, b], f'diagonal {a},{b}\n{pi}'
            else:
                assert pi[a, b] == 1 == pi[b, a], f'off diagonal {a},{b}\n{pi}'
                assert pi[a, a] == 0 == pi[b, b], f'diagonal {a},{b}\n{pi}'


    def _test_cyclic():
        cm = cyclic_matrix(8, 2)
        print(formatter.tostr(cm))


    def _test_xindexes():
        for _ in range(10):
            n = random.randint(10, 100)
            a = random.randrange(n)
            b = random.randrange(n)
            xs = xindexes(n, a, b)
            assert xs[a] == b and xs[b] == a


    _test_cyclic()
    _test_unitary()
    _test_2l()
    _test_xindexes()
    _test_permeye()
