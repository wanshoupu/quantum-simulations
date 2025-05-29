import random

from numpy import kron

from quompiler.construct.types import UnivGate
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.inter_product import inter_product, mykron
from quompiler.utils.mgen import random_unitary, random_su2

formatter = MatrixFormatter(precision=2)


def test_inter_product_reshaping():
    """
    given an inter_product, two different reshaping methods result in different matrix
    1. reshape(a,b,c,a,b,c)->transpose(0,3,1,4,2,5)->reshape(a*a,b*b,c*c)->transpose(0,2,1)
    2. reshape(a*b,c,a*b,c)->transpose(0,2,1,3)->reshape(a*b*a*b,c*c)
    """
    a, b, c = 2, 3, 2
    A = kron(np.arange(1, 1 + 2 ** 2).reshape(2, 2), np.ones((3, 3)))
    B = np.array([[1, 1j], [113, 113j]])
    M = inter_product(A, B, b)
    # print()
    # print(formatter.tostr(M))
    M1 = rearrange1(M, a, b, c)
    # print('rearrange1')
    # print(formatter.tostr(M1))

    M2 = rearrange2(M, a * b, c)
    # print('rearrange2')
    # print(formatter.tostr(M2))


def rearrange1(M, a, b, c):
    """
    reshape(a,b,c,a,b,c)->transpose(0,3,1,4,2,5)->reshape(a*a,b*b,c*c)->transpose(0,2,1)->reshape(a*a*c*c,b*b)
    """
    m2 = (M.reshape(a, b, c, a, b, c)
          .transpose(0, 3, 1, 4, 2, 5)
          .reshape(a * a, b * b, c * c)
          .transpose(0, 2, 1)
          .reshape(a * a * c * c, b * b))
    return m2


def rearrange2(M, a, c):
    """
    reshape(a,c,a,c)->transpose(0,2,1,3)->reshape(a*a,c*c)
    """
    m2 = np.reshape(M, (a, c, a, c)).transpose(0, 2, 1, 3).reshape(a * a, c * c)
    return m2


def test_inter_product_singlet_qubit():
    """
    given a unitary matrix of singlet qubit, study the inter_product of it with a number of eyes.
    """
    matrix = random_unitary(2)
    n = 2
    mats = [np.eye(2) for _ in range(n)]
    k = random.randrange(n)
    mats.insert(k, matrix)
    for m in mats:
        print()
        print(formatter.tostr(m))
    result = mykron(*mats)
    print()
    print(formatter.tostr(result))


def test_():
    # Example usage
    u = np.array([[0.15 - 0.51j, 0.51 - 0.68j],
                  [-0.51 - 0.68j, 0.15 + 0.51j]])

    V, W, u_reconstructed = construct_commutator_exact(u)

    print("Error (Frobenius norm):",
          np.linalg.norm(u - u_reconstructed))
    print("U:\n", u)
    print("Reconstructed U:\n", u_reconstructed)
    assert np.allclose(u, u_reconstructed)


import numpy as np
from scipy.linalg import expm


def construct_commutator_exact(u):
    # Step 1: Extract rotation angle theta from u
    cos_theta = np.real(np.trace(u) / 2)
    theta = np.arccos(np.clip(cos_theta, -1, 1))

    # Step 3: Construct v and w
    v = expm(1j * (theta / 2) * UnivGate.X.matrix)
    w = expm(1j * (theta / 2) * UnivGate.Y.matrix)

    # Step 4: Compute u' = v w v† w†
    u_reconstructed = v @ w @ v.conj().T @ w.conj().T

    return v, w, u_reconstructed
