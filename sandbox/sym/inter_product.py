from itertools import product

import sympy
from sympy import kronecker_product as kron


def validate_factors(factors):
    """
    Validate the relationship between m and factors and among the factors.
    Namely:
           a. f1 > f2 > f3 > ...
           b. f1 % f2 == 0, f2 % f3 == 0, ...
    If condition a. is violated, ValueError with message "Combine adjacent yeast (with same factor) together" will be raised;
    If condition b. is violated, ValueError with message "Invalid factors are detected and mesh_product cannot be carried out" will be raised.
    :param factors: list of int, (f1,f2,...) denotes the sizes of blocks to divide, bread, the original matrix into.
           seeds matrices will be mesh multiplied in Kronecker fashion.
           The size of factors == size of seeds and also m must be divisible by product(f0, f1, ...).
    """
    if any(f1 < f2 for f1, f2 in zip(factors, factors[1:])):
        raise ValueError(f'Some factors are inverted. Ensure factors are in non-decreasing order.')
    if any(f1 % f2 for f1, f2 in zip(factors, factors[1:])):
        raise ValueError(f'Some factor cannot equally divides its predecessor.')


def mesh_product(dough, yeast, factors):
    """
    This method is a shorthand for recursively applying `inter_product`:
    inter_product(inter_product(... inter_product(A, B, m), C, n), ...)

    It carries out the `inter_product` operation in a recursive, Big Endian fashion, where:
    - `dough = A` (the initial matrix),
    - `[B, C, ...]` are the `yeast` matrices to be applied,
    - `[m, n, ...]` are the corresponding `factors`.

    **Concept:**
    Yeast is applied to the dough in Big Endian order—i.e., the last yeast in the list affects the finest subdivision of the matrix dough.
    Visually:
        - -yeast0-yeast1- ... -yeast2- -yeast3-
    This requires that the factors shall be non-increasing order.
    If we divide A into equal blocks by factors, the block sizes will be the factors, f0, f1, ...
    We use notation A(f0), A(f1), ... to repr the factor matrices after this division.
    For more detail, see
    https://en.wikipedia.org/wiki/Kronecker_product#Tracy%E2%80%93Singh_product

    **Parameters:**
    :param dough: A square matrix of shape (m, m), where m > 0.
    :param yeast: A list of square matrices [Y1, Y2, ...] with shapes (k1, k1), (k2, k2), ..., where each ki > 0.
    :param factors: A list of integers [f1, f2, ...], where:
        - f1 divides m, f2 divides f1, and so on,
        - len(factors) == len(yeast),
        - The product of all factors must divide m exactly.

    Each factor determines how the matrix (bread) is recursively partitioned into smaller blocks. The corresponding yeast matrices are then applied using a Kronecker-style (mesh) multiplication to enrich the structure.

    **Returns:**
    The result of the recursive inter-product operation:
        A(f0) ⨁ Y1 ⨁ A(f1) ⨁ ... ⨁ Yn ⨁ A(fn)
    """

    validate_factors(factors)
    factor = 1
    for s, f in zip(yeast[::-1], factors[::-1]):
        dough = inter_product(dough, s, f * factor)
        factor *= s.shape[0]
    return dough


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
    :return: The inter product with A(n) ⨁ B ⨁ A(m) with N = n*m
    """
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    N = A.shape[0]
    assert len(B.shape) == 2 and B.shape[0] == B.shape[1]
    n = B.shape[0]

    # no change to A
    if n == 1:
        return A * B[0, 0]
    if N % m:
        raise ValueError(f'The dimension of A must be divisible by m but got dim={N} and m={m}')
    if m == 1:
        return kron(A, B)

    if N == m:
        return kron(B, A)

    C = sympy.zeros(N * n)
    for i, j in product(range(0, N, m), range(0, N, m)):
        for k, l in product(range(n), range(n)):
            C[n * i + k * m:n * i + (k + 1) * m, n * j + l * m:n * j + (l + 1) * m] = A[i:i + m, j:j + m] * B[k, l]
    return C
