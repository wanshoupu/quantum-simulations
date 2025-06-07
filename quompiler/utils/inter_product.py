from itertools import chain
from typing import Sequence

import numpy as np
from numpy import kron
from numpy.typing import NDArray

from quompiler.utils.mfun import unitary_prop, id_prop

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


def normalize(factors: Sequence[NDArray]) -> list[NDArray]:
    """
    Clean up and normalize the factors such that identity matrices, unitary matrices would be made so as much as it can.
    Detect matrices that's proportional to unitary matrices and scale them to make them unitary while preserving the overall product unchanged.
    :param factors: matrix factors.
    :return: Make as many factors unitary as possible and return the scaled factors. Otherwise, return factors as is.
    """
    length = len(factors)
    assert length > 1

    factors = list(factors)
    scale = 1
    nonids = []
    for i in range(length):
        p = id_prop(factors[i])
        if p:
            factors[i] = factors[i] / p.ratio
            scale *= p.ratio
        else:
            nonids.append(i)
    dump_idxs = []
    for i in nonids:
        p = unitary_prop(factors[i])
        if p:
            factors[i] = factors[i] / p.ratio
            scale *= p.ratio
        else:
            dump_idxs.append(i)
    dump_idx = dump_idxs[0] if dump_idxs else (nonids[0] if nonids else 0)
    factors[dump_idx] *= scale

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
            return normalize([A, B])
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
    For example, with a block size of 2x2
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
                return normalize([A, B]), [c]
    kf = kron_factor(M)
    return kf, [1] * (len(kf) - 1)


def is_idler(mat: NDArray, idx: int) -> bool:
    """
    Check if a square matrix, deemed as multi-qubit operator, is an idler for the qubit at index `idx`.
    :param mat: a square matrix of shape (N, N) where N = 1<<n.
    :param idx:
        The index (0-based) of the qubit subsystem to be projected out, where |ψ⟩ lives.
        Note that our matrix is based on [qubit-0 ⨂ qubit-1 ⨂ ...].
    :return: True if the given qubit at index `idx` is an idler, otherwise False.
    """
    N = mat.shape[0]
    assert mat.shape == (N, N) and 4 <= N and N & (N - 1) == 0, f" {mat} must be a square matrix of order of power of 2 and order>=4"
    n = N.bit_length() - 1
    assert 0 <= idx < n, f"idx {idx} must be within range(0, {n})"
    even_idxs = [i for i in range(N) if i & (1 << (n - 1 - idx)) == 0]
    odd_idxs = [i for i in range(N) if i & (1 << (n - 1 - idx)) != 0]
    sub_ee = mat[np.ix_(even_idxs, even_idxs)]
    sub_eo = mat[np.ix_(even_idxs, odd_idxs)]
    sub_oe = mat[np.ix_(odd_idxs, even_idxs)]
    sub_oo = mat[np.ix_(odd_idxs, odd_idxs)]
    return np.allclose(sub_oo, sub_ee) and np.allclose(sub_eo, np.zeros_like(sub_eo)) and np.allclose(sub_oe, np.zeros_like(sub_oe))


def qproject(mat: NDArray, idx: int, state: NDArray) -> NDArray:
    """
    Project a square matrix, deemed as multi-qubit operator, onto a subsystem where the target qubit is fixed in a given pure state |ψ⟩.

    This computes the effective operator on the remaining qubits by sandwiching U with |ψ⟩ on the specified qubit.
    That is, it returns:
        U_eff = ⟨ψ| U |ψ⟩

    Parameters:
    -----------
    :param mat: a square matrix of shape (N, N) where N = 1<<n.
    :param state: np.ndarray, state to be used to project the qubit.
        A 2-dimensional complex vector representing a normalized pure state |ψ⟩ of a single qubit.

    :param idx:
        The index (0-based) of the qubit subsystem to be projected out, where |ψ⟩ lives.
        Note that our matrix is based on [qubit-0 ⨂ qubit-1 ⨂ ...].

    Returns:
    --------
    U_eff : NDArray
        A NDArray representing the projected operator acting on the remaining n-1 qubits.

    Notes:
    ------
    - This is not the same as a partial trace. It conditions on the subsystem being in a known state |ψ⟩.
    - Useful for computing effective dynamics or gates when a subsystem is initialized or post-selected in a known pure state.

    Example:
    --------
     ψ = np.array([1, 0])  # |0>
     U = np.array(...)  # |01>
     U_eff = U.project(ψ, qubit=0)
    """
    assert state.shape == (2,), f'state vector must be a 1D array of length 2, but got {state.shape}'
    N = mat.shape[0]
    assert mat.shape == (N, N) and 4 <= N and N & (N - 1) == 0, f" {mat} must be a square matrix of order of power of 2 and order>=4"
    n = N.bit_length() - 1
    assert 0 <= idx < n, f"idx {idx} must be within range(0, {n})"
    idxs0 = [i for i in range(N) if i & (1 << (n - 1 - idx)) == 0]
    idxs1 = [i for i in range(N) if i & (1 << (n - 1 - idx)) != 0]
    a = np.conj(state[0]) * state[0] * mat[np.ix_(idxs0, idxs0)]
    b = np.conj(state[1]) * state[0] * mat[np.ix_(idxs1, idxs0)]
    c = np.conj(state[0]) * state[1] * mat[np.ix_(idxs0, idxs1)]
    d = np.conj(state[1]) * state[1] * mat[np.ix_(idxs1, idxs1)]
    return a + b + c + d


def ctrl_expand(A, block_size, active):
    """
    Insert control effect into the matrix A.
    For example, given A=
    ⎡a₀₀  a₀₁  a₀₂  a₀₃⎤
    ⎢                  ⎥
    ⎢a₁₀  a₁₁  a₁₂  a₁₃⎥
    ⎢                  ⎥
    ⎢a₂₀  a₂₁  a₂₂  a₂₃⎥
    ⎢                  ⎥
    ⎣a₃₀  a₃₁  a₃₂  a₃₃⎦

    block_ctrl(A, 2, 1) =

    ⎡1  0   0    0   0  0   0    0 ⎤
    ⎢                              ⎥
    ⎢0  1   0    0   0  0   0    0 ⎥
    ⎢                              ⎥
    ⎢0  0  a₀₀  a₀₁  0  0  a₀₂  a₀₃⎥
    ⎢                              ⎥
    ⎢0  0  a₁₀  a₁₁  0  0  a₁₂  a₁₃⎥
    ⎢                              ⎥
    ⎢0  0   0    0   1  0   0    0 ⎥
    ⎢                              ⎥
    ⎢0  0   0    0   0  1   0    0 ⎥
    ⎢                              ⎥
    ⎢0  0  a₂₀  a₂₁  0  0  a₂₂  a₂₃⎥
    ⎢                              ⎥
    ⎣0  0  a₃₀  a₃₁  0  0  a₃₂  a₃₃⎦

    block_ctrl(A, 2, 0) =

    ⎡a₀₀  a₀₁  0  0  a₀₂  a₀₃  0  0⎤
    ⎢                              ⎥
    ⎢a₁₀  a₁₁  0  0  a₁₂  a₁₃  0  0⎥
    ⎢                              ⎥
    ⎢ 0    0   1  0   0    0   0  0⎥
    ⎢                              ⎥
    ⎢ 0    0   0  1   0    0   0  0⎥
    ⎢                              ⎥
    ⎢a₂₀  a₂₁  0  0  a₂₂  a₂₃  0  0⎥
    ⎢                              ⎥
    ⎢a₃₀  a₃₁  0  0  a₃₂  a₃₃  0  0⎥
    ⎢                              ⎥
    ⎢ 0    0   0  0   0    0   1  0⎥
    ⎢                              ⎥
    ⎣ 0    0   0  0   0    0   0  1⎦

    :param A: square matrix
    :param block_size: the block size to be inserted.
    :param active:
    :return:
    """

    def ctr_prod(block, spacer):
        zero = np.zeros_like(block)
        if active:
            return np.block([[spacer, zero], [zero, block]])
        return np.block([[block, zero], [zero, spacer]])

    N = A.shape[0]
    assert A.shape == (N, N)
    assert N % block_size == 0, "block_size must divide matrix size evenly"

    blocks = []
    eye = np.eye(block_size)
    zero = np.zeros_like(eye)
    for i in range(0, N, block_size):
        b = [ctr_prod(A[i:i + block_size, j:j + block_size], eye if i == j else zero) for j in range(0, N, block_size)]
        blocks.append(b)

    return np.block(blocks)
