import numpy as np
from numpy._typing import NDArray
from scipy.linalg import eig

from quompiler.construct.types import UnivGate
from quompiler.utils.mfun import herm


def gc_decompose(U: NDArray) -> tuple[NDArray, NDArray]:
    """
    Group commutator decomposition. This is exact decomposition with no approximation.
    Given a SU(2) operator, representing rotation R(n, φ), we decompose it into ±V @ W @ V† @ W†,
    with V and W rotations of angle θ around x-axis and y-axis, respectively.
    The plusminus sign comes from trace-negative / trace-positive operators.
    :param U: input 2x2 unitary matrix.
    :return: tuple(V, W, sign) such that U = V @ W @ V† @ W† * sign
    """
    alpha = rangle(U)
    beta = 2 * np.arcsin(np.sqrt(np.cos(alpha / 4)))
    # assert np.isclose(np.sin(alpha / 2), 2 * np.sin(beta / 2) ** 2 * np.sqrt(1 - np.sin(beta / 2) ** 4))

    W = UnivGate.X.rotation(beta)
    V = UnivGate.Y.rotation(beta)
    U2 = V @ W @ herm(V) @ herm(W)

    P = tsim(U2, U)
    WA = P @ W @ herm(P)
    VA = P @ V @ herm(P)
    return VA, WA


def tsim(U, V):
    """
   Compute the similarity transformation matrix P such that V = P @ U @ P†, assuming such a transformation exists.

    This function does not verify whether U and V are actually similar; it simply
    attempts to compute a matrix P that satisfies the relation if possible.

    Parameters:
        U (np.ndarray): A 2×2 unitary matrix (e.g., an SU(2) operator).
        V (np.ndarray): A 2×2 unitary matrix (e.g., an SU(2) operator).

    Returns:
        np.ndarray: A 2×2 unitary matrix P such that V = P @ U @ P†.

    Note:
        The function assumes U and V are both unitary, but does not check or enforce this.
        The result may not be valid if U and V are not similar (i.e., do not have the same eigenvalues).
    """
    _, uval, uvec = eigen_decompose(U)
    _, vval, vvec = eigen_decompose(V)
    X = np.array(UnivGate.X)
    if np.allclose(uval, vval[::-1]):
        # swap the eigen values along with the eigenvectors
        return vvec @ X @ herm(uvec)
    return vvec @ herm(uvec)


def eigen_decompose(U: NDArray) -> tuple[np.complex128, NDArray, NDArray]:
    """
    Compute the eigenvalues and eigenvectors of a unitary matrix such that U V = V D
    :param U:
    :return: phase, D, V such that U = V @ D @ V† * phase
    """
    phase = gphase(U)
    val, vec = eig(U / phase)
    return phase, val, vec


def rangle(U: NDArray) -> float:
    """
    Rotation angle for the input SU(2) operator around some unit vector n.
    This angle so calculated is between 0 and π such that the global phase takes care of the negative-trace.
    :param U: SU(2) operator
    :return: the rotation angle (alpha)
    """
    trace = np.trace(U) / gphase(U)
    arg = np.real(trace) / 2
    return 2 * np.arccos(np.clip(arg, -1.0, 1.0))


def raxis(U: NDArray) -> NDArray:
    """
    calculate the rotation axis of SU(2) operator around some unit vector n.
    :param U:
    :return: unit vector n
    """
    U = U / gphase(U)
    angle = rangle(U)
    if np.equal(angle, 0):
        return np.array([0, 0, 1])
    V = 1j * (U - np.cos(angle / 2) * np.eye(2)) / np.sin(angle / 2)
    # assert np.isclose(np.trace(V), 0)
    # assert np.isclose(np.imag(V[0, 0]), 0)
    x, y = np.real(V[1, 0]), np.imag(V[1, 0])
    z = np.clip(-np.real(V[1, 1]), -1, 1)
    return np.array([x, y, z])


def euler_params(u: NDArray) -> tuple[complex, float, float, float]:
    """
    Given a U(2) matrix, decompose it into Euler angles + an overall scalar factor.
    The ranges of the parameters are:
    abs(a) == 1
    -π < b < π
    -π < c < π
    -π < d < π
    :param u: U(2) matrix as input
    :return: scalar factor + Euler angles (b, c, d), such that u = a * Rz(b) @ Ry(c) @ Rz(d)
    """
    assert u.shape == (2, 2) and np.allclose(u.conj().T @ u, np.eye(2)), "u must be unitary"
    det = np.linalg.det(u)
    c2 = u[0, 0] * u[1, 1] / det
    s2 = -u[1, 0] * u[0, 1] / det
    # assert np.isclose(1, c2 + s2)
    plus = c2 if np.isclose(c2, 0) else np.angle(u[1, 1] / u[0, 0])
    minus = s2 if np.isclose(s2, 0) else np.angle(-u[1, 0] / u[0, 1])

    b = (plus + minus) / 2  # angle pre-Y rotation
    d = (plus - minus) / 2  # angle post-Y rotation

    # Determine rotation angle around Y (theta)
    x = np.real_if_close(c2 - s2)
    y = np.real_if_close(2 * u[1, 0] * u[1, 1] / det / np.exp(1j * b))
    # assert np.isclose(x.imag, 0) and np.isclose(y.imag, 0)
    c = np.arctan2(y, x)
    # Global phase
    a = (u[1, 1] / (np.cos(c / 2) * np.exp(.5j * (b + d)))) if c2 > s2 else (u[1, 0] / (np.sin(c / 2) * np.exp(.5j * (b - d))))
    return a, b, c, d  # Global phase (a), and Euler angles (b, c, d)


def rota(theta: float, phi: float, alpha: float) -> NDArray:
    """
    Generate a SU2 unitary matrix based on the rotation of angle α around the axis prescribed by the spherical coordinate (θ, φ).
    U =
    :param theta: polar coordinate of the axis
    :param phi: azimuthal coordinate of the axis
    :param alpha: rotation angle.
    :return: SU2 unitary matrix
    """
    nvec = np.sin(theta) * np.cos(phi), np.cos(theta) * np.cos(phi), np.cos(theta)
    return rot(np.array(nvec), alpha)


def rot(n: NDArray, angle: float) -> NDArray:
    """
    Generate a SU2 unitary matrix based on the rotation angle around the axis prescribed by the vector in 3D Euclidean space.
    :param n: the vector in 3D Euclidean space given by (x,y,z).
    :param angle: the rotation angle.
    :return: SU2 unitary matrix
    """
    assert n.shape == (3,)
    n = n / np.linalg.norm(n)
    pvec = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    sigma = np.tensordot(n, pvec, axes=1)
    u = np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * sigma
    return u


def dist(u: NDArray, v: NDArray) -> float:
    """
    Compute the trace distance between two unitary matrices of shape (2,2), e.g., for a single qubit.
    Distance of two unitary matrices is defined as
    1. calculate the product Δ = u @ v†;
    2. Model Δ as a rotation around certain axis and calculate the rotation angle θ ∈ [-π, π];
    3. D(u,v) = 2 abs(sin(θ/4)).

    The canonical definition of distance is cumbersome in practice, so we simplified it for ease of calculation.
    We adopted a simplified version, dist = 4 (sin(θ/4))^2.
    :param u: unitary matrix.
    :param v: another unitary matrix.
    :return: trace distance as defined.
    """
    assert u.shape == v.shape == (2, 2), "operators must have shape (2, 2)"
    delta = u @ herm(v)
    # assert np.allclose(delta @ herm(delta), np.eye(2)), "matmul product must be unitary"
    ct = np.trace(delta) / gphase(delta)
    # sometimes the result is slightly complex. We take the real part
    # assert np.isclose(ct.imag, 0)
    return 2 - np.real(ct)


def gphase(u: NDArray) -> np.complex128:
    """
    Calculate the global phase of unitary matrix such that
        up = np.conj(gp) * u
    will be a positively traced matrix with unit determinant.
    :param u: unitary matrix
    :return:
    """
    det = np.linalg.det(u)
    # unit-magnitude complex number (e^{iϕ})
    phase = np.sqrt(det, dtype=np.complex128)
    if np.trace(u / phase) < 0:
        return -phase
    return phase


def vec(U: NDArray) -> tuple[float, float, float]:
    """
    Vectorized version of unitary matrix.
    Given a unitary operator, calculate its unique representation as a 4D vector of real numbers: (θ, φ, α, β)
    such that u = cos(α/2) * I - i * sin(α / 2) * [X * sin(θ) * cos(φ) + Y * sin(θ) * sin(φ) + Z * cos(θ)],
    where θ is the polar angle, φ is the azimuth angle, α is the angle of rotation, and β is the phase angle.
    Their ranges are as follows:
    0 <= θ < π
    -π <= φ < π
    0 <= α < π
    0 <= β < 2 * π
    :param U: unitary matrix
    :return: A vector of real numbers (θ, φ, α). The phase factor is left out as it is non-essential.
    """
    U = U / gphase(U)
    alpha = rangle(U)
    if np.equal(alpha, 0):
        return 0, 0, 0
    V = 1j * (U - np.cos(alpha / 2) * np.eye(2)) / np.sin(alpha / 2)
    x, y = np.real(V[1, 0]), np.imag(V[1, 0])
    z = -np.real(V[1, 1])
    return np.arccos(np.clip(z, -1, 1)), np.arctan2(y, x), alpha


def vec4su2net(U: NDArray) -> tuple[float, float, float]:
    """
    This is a simplified version of `vec` whose sole purpose is for SU2Net. Difference between the two are the scaled theta and phi because
    1. as alpha -> 0 or π, the significance of the spherical params decrease.
    2. as theta -> 0 or π, the significance of the φ decreases.
    """
    U = U / gphase(U)
    cosa = np.real(np.trace(U)) / 2
    alpha = 2 * np.arccos(np.clip(cosa, -1.0, 1.0))
    if np.isclose(cosa, 1):
        return 0, 0, alpha
    V = 1j * (U - np.cos(alpha / 2) * np.eye(2)) / np.sin(alpha / 2)
    x, y = np.real(V[1, 0]), np.imag(V[1, 0])
    z = -np.real(V[1, 1])
    return cosa * np.arccos(np.clip(z, -1, 1)), cosa * z * np.arctan2(y, x), alpha


def mod_dist(x, y):
    """
    I hope to use this as a topological metric to help improve the SU2Net lookup accuracy.
    :param x:
    :param y:
    :return:
    """
    d = np.abs(x - y)
    d2 = np.array([d[0], 2 * np.pi - d[1], d[2]])
    d3 = np.array([np.pi - d[0], (d[1] + np.pi) % (2 * np.pi), np.pi - d[2]])
    return np.minimum(sum(d), sum(d2), sum(d3))
