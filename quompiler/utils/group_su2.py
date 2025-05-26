from numpy._typing import NDArray


def gc_decompose(U: NDArray) -> tuple[NDArray, NDArray]:
    """
    Group commutator decomposition. This is exact decomposition with no approximation.
    Given a SU(2) operator, representing rotation R(n, φ), we decompose it into V @ W @ V† @ W†,
    with V and W rotations of angle θ around x-axis and y-axis, respectively.
    :param U: input 2x2 unitary matrix.
    :return: tuple(V, W) such that U = V @ W @ V† @ W†
    """

    pass
