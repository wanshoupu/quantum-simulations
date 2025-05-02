from itertools import chain
from typing import Sequence

import numpy as np
from numpy import kron
from numpy.typing import NDArray

from quompiler.utils.mfun import allprop

ONE = np.eye(1)


def mykron(*matrices: NDArray) -> NDArray:
    assert len(matrices) > 0
    if len(matrices) == 1:
        return matrices[0]
    return mykron(kron(matrices[0], matrices[1]), *matrices[2:])


def validm(m: NDArray):
    s = m.shape
    if len(s) != 2:
        raise ValueError(f'Matrix must be 2D array but got {s}.')
    if s[0] != s[1]:
        raise ValueError(f'Matrix must be square but got {s}.')


def int_factors(n: int) -> Sequence[tuple[int, int]]:
    result = []
    for i in range(2, n):
        if n % i == 0:  # Check if `i` divides n
            j = n // i
            result.append((i, j))
    return result


def clean_id(*factors):
    """
    Detect matrices that's proportional to identities and scale them to make them identities while preserving the overall product unchanged.
    :param factors: matrix factors.
    :return: Make as many factors identity as possible and return the scaled factors. Otherwise, return factors as is.
    """
    assert len(factors) > 1
    data = [allprop(m, np.eye(m.shape[0])) + (m,) for m in factors]
    dump = next((i for i, t in enumerate(data) if not t[0]), 0)
    factors = list(factors)
    for i in range(len(data)):
        y, r, m = data[i]
        if y:
            factors[i] /= r
            factors[dump] *= r
    return factors


def kron_factor(M: NDArray) -> list[NDArray]:
    validm(M)
    m = M.shape[0]
    for a, b in int_factors(m):
        M2 = block_flatten(M, a, b)

        # Step 4: SVD
        U, S, Vh = np.linalg.svd(M2)

        # Step 5: Take first singular value/vector
        u = U[:, 0]
        v = Vh[0, :]
        s = S[0]

        A = np.sqrt(s) * u.reshape(a, a)
        B = np.sqrt(s) * v.reshape(b, b)

        if np.allclose(mykron(A, B), M):
            return list(clean_id(A, B))
    return [M]


def recursive_kron_factor(M: NDArray) -> list[NDArray]:
    factors = kron_factor(M)
    if len(factors) == 1:
        return factors
    return list(chain.from_iterable(recursive_kron_factor(m) for m in factors))  # flatten


def block_flatten(M, m, n):
    """
    Treating input matrix as a block matrix, with block size (n x n), flatten out each block into a row with a total of m x m rows
    :param M: a square matrix of shape (m x n, m x n)
    :param m: int, a factor of S
    :param n: int, must be the quotient S / m
    :return: a rectangular matrix of shape (m ** 2, n ** 2).
    """
    # Reshape into 4D tensor (block matrices bmat)
    bmat = M.reshape((m, n, m, n))
    # flatten with dimension (m ** 2, n ** 2)
    M2 = bmat.transpose(0, 2, 1, 3).reshape(m ** 2, n ** 2)
    return M2


def block_square(M, m, n):
    """
    Treating input matrix as a flattened block matrix with shape (m ** 2, n ** 2), convert it to explicit square matrix of shape (m x n, m x n)
    :param M: a rectangular matrix of shape (m ** 2, n ** 2)
    :param m: int, representing the number of blocks, must be an integer factor of S
    :param n: int, representing the block size, must be the quotient S / m
    :return: a square matrix of shape (m x n, m x n)
    """
    # Reshape into 4D tensor (block matrices bmat)
    bmat = M.reshape((m, m, n, n))
    # flatten with shape (m * n, m * n)
    return bmat.transpose(0, 2, 1, 3).reshape(m * n, m * n)


def validate_factors(factors):
    """
    Validate the relationship between m and factors and among the factors.
    Namely:
           a. 1 <= f1 <= f2 <= f3 <= ... <= n
           b. f2 % f1 == 0, f3 % f2 == 0, ...
    :param factors: list of int, (f1,f2,...) denotes the number of blocks to divide matrix into to perform the Tracy-Singh product successively.
           The size of factors == size of seeds and also m must be divisible by product(f0, f1, ...).
    """
    if any(f2 % f1 for f1, f2 in zip(factors, factors[1:])):
        raise ValueError(f'A factor must divide its respective predecessor.')


def mesh_product(matrices: Sequence[NDArray], partitions: Sequence[int]):
    """
    This method is a shorthand for successively applying `inter_product` between the first two matrices in matrices:
    inter_product(inter_product(inter_product(A, B, m), C, n), ...) where matrices = [B, C, ...] and partitions = [m, n, ...].

    **Concept:**
    This is a special case of Tracy-Singh product. For more detail, see
    https://en.wikipedia.org/wiki/Kronecker_product#Tracy%E2%80%93Singh_product
    An analogy is made below for this mathematical operation:
    - The `dough` the first element in matrices, start with matrices[0]
    - The `yeast` the subsequent element in matrices. Yeast is to be ⨂ onto dough in successively.
    - The `meshing partition` the number of rows and columns to partition the dough into, namely, m x m equal-sized blocks.
    It is required that the partitions shall satisfy the hierarchical relationship. See :param partitions.
    We use notation A(m), A(n), ... to repr the block matrices of size m x m, n x n, ...

    **Parameters:**
    :param matrices: A list of square matrices with shapes (k1, k1), (k2, k2), ..., where each ki > 0.
    :param partitions: A list of integers [f0, f1, ...], representing the hierarchical meshing partitions of the matrix, dough, into blocks.
    The partitions must satisfy the following conditions:
        - partitions start coarse and become finder and finer, namely, m <= n <= ...
        - partitions are nested, namely, m % 1 == n % m == ... == z % dough.shape[0] == 0.
        - len(partitions) == len(matrices)

    **Returns:**
    The result of the recursive inter-product operation:
        A(m) ⨂ Yi ⨂ A(n) ⨂ ... ⨂ Yj ⨂ A(z)
    """
    assert len(matrices) == len(partitions)
    assert partitions[-1] == matrices[0].shape[0]
    validate_factors(partitions)
    dough = matrices[0]
    N = dough.shape[0]
    for s, f in zip(matrices[1:], partitions):
        dough = inter_product(dough, s, N // f)
    return dough


def mesh_factor(M: NDArray) -> tuple[list[NDArray], list[int]]:
    """
    This is a reverse function of mesh_product: it factors mesh_product into its factor matrices and rising factors.
    :param M: a square matrix to be factored.
    :return: A list of factor matrices and their meshing partitions according to the mesh_product rule.
    The two list should be of same size. The last number in the meshing partitions is the dough shape[0].
    When there is no way to factor M, return [M], [M.shape[0]].
    """
    ms, factors = inter_factor(M)
    if not factors:
        return [M], [M.shape[0]]
    dough, yeast = ms
    ms2, factor2 = mesh_factor(dough)
    dough2, yeast2 = ms2[0], ms2[1:]
    return [dough2, yeast] + yeast2, [dough.shape[0] // factors[0]] + factor2


def inter_product(A, B, m):
    """
    A matrix multiplication operation of special Tracy–Singh product type.
    https://en.wikipedia.org/wiki/Kronecker_product#Tracy%E2%80%93Singh_product
    Matrix A is first divided up into n x n number of m x m sized blocks;
    Then B is mesh multiplied between the n x n number of m x m sized blocks.
    For example,
        A =
        ⎡a₀₀  a₀₁  a₀₂  a₀₃⎤
        ⎢                  ⎥
        ⎢a₁₀  a₁₁  a₁₂  a₁₃⎥
        ⎢                  ⎥
        ⎢a₂₀  a₂₁  a₂₂  a₂₃⎥
        ⎢                  ⎥
        ⎣a₃₀  a₃₁  a₃₂  a₃₃⎦
        B =
        ⎡b₀₀  b₀₁⎤
        ⎢        ⎥
        ⎣b₁₀  b₁₁⎦
        inter_product(A, B) =
        ⎡a₀₀⋅b₀₀  a₀₁⋅b₀₀  a₀₀⋅b₀₁  a₀₁⋅b₀₁  a₀₂⋅b₀₀  a₀₃⋅b₀₀  a₀₂⋅b₀₁  a₀₃⋅b₀₁⎤
        ⎢                                                                      ⎥
        ⎢a₁₀⋅b₀₀  a₁₁⋅b₀₀  a₁₀⋅b₀₁  a₁₁⋅b₀₁  a₁₂⋅b₀₀  a₁₃⋅b₀₀  a₁₂⋅b₀₁  a₁₃⋅b₀₁⎥
        ⎢                                                                      ⎥
        ⎢a₀₀⋅b₁₀  a₀₁⋅b₁₀  a₀₀⋅b₁₁  a₀₁⋅b₁₁  a₀₂⋅b₁₀  a₀₃⋅b₁₀  a₀₂⋅b₁₁  a₀₃⋅b₁₁⎥
        ⎢                                                                      ⎥
        ⎢a₁₀⋅b₁₀  a₁₁⋅b₁₀  a₁₀⋅b₁₁  a₁₁⋅b₁₁  a₁₂⋅b₁₀  a₁₃⋅b₁₀  a₁₂⋅b₁₁  a₁₃⋅b₁₁⎥
        ⎢                                                                      ⎥
        ⎢a₂₀⋅b₀₀  a₂₁⋅b₀₀  a₂₀⋅b₀₁  a₂₁⋅b₀₁  a₂₂⋅b₀₀  a₂₃⋅b₀₀  a₂₂⋅b₀₁  a₂₃⋅b₀₁⎥
        ⎢                                                                      ⎥
        ⎢a₃₀⋅b₀₀  a₃₁⋅b₀₀  a₃₀⋅b₀₁  a₃₁⋅b₀₁  a₃₂⋅b₀₀  a₃₃⋅b₀₀  a₃₂⋅b₀₁  a₃₃⋅b₀₁⎥
        ⎢                                                                      ⎥
        ⎢a₂₀⋅b₁₀  a₂₁⋅b₁₀  a₂₀⋅b₁₁  a₂₁⋅b₁₁  a₂₂⋅b₁₀  a₂₃⋅b₁₀  a₂₂⋅b₁₁  a₂₃⋅b₁₁⎥
        ⎢                                                                      ⎥
        ⎣a₃₀⋅b₁₀  a₃₁⋅b₁₀  a₃₀⋅b₁₁  a₃₁⋅b₁₁  a₃₂⋅b₁₀  a₃₃⋅b₁₀  a₃₂⋅b₁₁  a₃₃⋅b₁₁⎦
    :param A: square matrix of shape (N, N).
    :param B: square matrix of shape (k, k), with k > 0.
    :param m: an integer factor of N, denotes the size of blocks to divide the matrix A into.
    :return: The inter product with A(n) ⨂ B ⨂ A(m) with N = n*m
    """
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    N = A.shape[0]
    assert len(B.shape) == 2 and B.shape[0] == B.shape[1]
    if B.shape[0] == 1:
        return A * B[0, 0]
    if N % m:
        raise ValueError(f'The dimension of A must be divisible by m but got N={N} and m={m}')
    if m == 1:
        return kron(A, B)

    if N == m:
        return kron(B, A)

    blocks = []
    for i in range(0, N, m):
        b = [kron(B, A[i:i + m, j:j + m]) for j in range(0, N, m)]
        blocks.append(b)

    return np.block(blocks)


def inter_factor(M: NDArray) -> tuple[list[NDArray], list[int]]:
    """
    This is a reverse function of inter_product: it factors inter_product into its factor matrices, if any.
    :param M: a square matrix to be factored.
    :return: A list of factor matrices and their meshing size according to the mesh_product rule.
    The two list should be of same size. The last number in the meshing size is the dough shape[0].
    When there is no way to factor M, return [M], [M.shape[0]].
    :return: a tuple(list of factor matrices and list of block sizes) according to the inter_product rule.
    The length of the factor matrices is either 1 and 2. The block size is either empty or contains one int.
    When block size is empty, there should be exactly one factor which is M itself.
    """
    validm(M)
    m = M.shape[0]
    for a, bc in int_factors(m):
        for b, c in int_factors(bc):
            # these manipulation of the matrix is to separate the block divisions during the inter_product
            # Namely, M is block divided into a x a blocks of shape(c,c) and then multiplied by B of shape(b,b) in the inter_product style
            M2 = (M.reshape(a, b, c, a, b, c)
                  .transpose(0, 3, 1, 4, 2, 5)
                  .reshape(a * a, b * b, c * c)
                  .transpose(0, 2, 1)
                  .reshape(a * a * c * c, b * b))
            # SVD to attempt to factor the yeast matrix
            U, S, Vh = np.linalg.svd(M2)

            # Take largest singular value/vector
            u = U[:, 0]
            v = Vh[0, :]
            s = S[0]

            A = np.sqrt(s) * block_square(u, a, c)
            B = np.sqrt(s) * v.reshape(b, b)

            if np.allclose(inter_product(A, B, c), M):
                return list(clean_id(A, B)), [c]
    kf = kron_factor(M)
    return kf, [1] * (len(kf) - 1)
