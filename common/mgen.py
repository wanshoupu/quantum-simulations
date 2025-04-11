import numpy as np
import random



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


def random_matrix_2l(n, r1, r2):
    random.seed(3)
    rr = lambda: random.randint(0, 10)
    m = np.diag([1 + 0j] * n)
    r1, r2 = min(r1, r2), max(r1, r2)
    m[r1, r1] = complex(rr(), rr())
    m[r2, r1] = complex(rr(), rr())
    m[r1, r2] = complex(rr(), rr())
    m[r2, r2] = complex(rr(), rr())
    return m


if __name__ == '__main__':
    from common.format_matrix import MatrixFormatter

    formatter = MatrixFormatter()
    m2l = random_matrix_2l(10, 1, 6)
    print(formatter.tostr(m2l))

    randu = random_unitary(2)
    print(formatter.tostr(randu))
    print(formatter.tostr(randu.T @ np.conj(randu)))
