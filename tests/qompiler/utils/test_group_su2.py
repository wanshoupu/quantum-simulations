import random

import numpy as np
import pytest

from quompiler.construct.types import UnivGate
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.group_su2 import rangle, gc_decompose, tsim, raxis, rot, euler_params, gphase, dist, eigen_decompose, vec, mod_dist, rota
from quompiler.utils.mfun import herm, allprop
from quompiler.utils.mgen import random_unitary, random_su2, random_phase

formatter = MatrixFormatter(precision=2)


@pytest.mark.parametrize('gate,expected', [
    [UnivGate.I, (1, 0, 0, 0)],
    [UnivGate.X, (-1j, np.pi / 2, np.pi, -np.pi / 2)],
    [UnivGate.Y, (-1j, 0, -np.pi, 0)],
    [UnivGate.Z, (1j, np.pi / 2, 0, np.pi / 2)],
    [UnivGate.H, (1j, np.pi, -np.pi / 2, 0)],
    [UnivGate.S, (np.sqrt(1j), np.pi / 4, 0, np.pi / 4)],
    [UnivGate.T, (np.power(1j, 1 / 4), np.pi / 8, 0, np.pi / 8)],
])
def test_euler_params_std_gate(gate: UnivGate, expected: tuple):
    coms = euler_params(gate.matrix)
    a, b, c, d = coms
    actual = a * UnivGate.Z.rotation(b) @ UnivGate.Y.rotation(c) @ UnivGate.Z.rotation(d)
    assert np.allclose(actual, gate.matrix)
    assert np.allclose(coms, expected)


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 100))
def test_euler_params_verify_identity_random(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    expected = random_unitary(2)
    a, b, c, d = euler_params(expected)
    actual = a * UnivGate.Z.rotation(b) @ UnivGate.Y.rotation(c) @ UnivGate.Z.rotation(d)
    assert np.allclose(actual, expected)


@pytest.mark.parametrize("gate,phase,expected", [
    [UnivGate.I, 1, 1],
    [UnivGate.I, -1, -1],
    [UnivGate.I, 1j, 1j],
    [UnivGate.I, -1j, -1j],
    [UnivGate.X, 1, 1j],
    [UnivGate.X, -1, 1j],
    [UnivGate.X, 1j, 1],
    [UnivGate.X, -1j, 1],
    [UnivGate.Y, 1, 1j],
    [UnivGate.Y, -1, 1j],
    [UnivGate.Y, 1j, 1],
    [UnivGate.Y, -1j, 1],
    [UnivGate.Z, 1, 1j],
    [UnivGate.Z, -1, 1j],
    [UnivGate.Z, 1j, 1],
    [UnivGate.Z, -1j, 1],
    [UnivGate.H, 1, 1j],
    [UnivGate.H, -1, 1j],
    [UnivGate.H, 1j, 1],
    [UnivGate.H, -1j, 1],
    [UnivGate.S, 1, np.exp(1j * np.pi / 4)],
    [UnivGate.S, -1, -np.exp(1j * np.pi / 4)],
    [UnivGate.S, 1j, 1j * np.exp(1j * np.pi / 4)],
    [UnivGate.S, -1j, -1j * np.exp(1j * np.pi / 4)],
    [UnivGate.T, 1, np.exp(1j * np.pi / 8)],
    [UnivGate.T, -1, -np.exp(1j * np.pi / 8)],
    [UnivGate.T, 1j, 1j * np.exp(1j * np.pi / 8)],
    [UnivGate.T, -1j, -1j * np.exp(1j * np.pi / 8)],
])
def test_gphase_std(gate: UnivGate, phase, expected):
    """
    There is a special phase 1j on standard UnivGate that demands special attention.
    Instead of being zero, the distance is 4.
    """
    u = np.array(gate) * phase
    # execute
    actual = gphase(u)
    # verify
    assert np.isclose(actual, expected), f"{actual} != {expected}"
    assert np.isclose(np.linalg.det(u / gphase(u)), 1)
    assert np.trace(u / actual) >= 0


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 100))
def test_gphase_random(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    u = random_unitary(2)
    # execute
    phase = gphase(u)
    # verify
    assert np.isclose(np.linalg.det(u / phase), 1)
    assert np.trace(u / phase) >= 0


@pytest.mark.parametrize("gate,expected", [
    [UnivGate.I, 0],
    [UnivGate.X, np.pi],
    [UnivGate.Y, np.pi],
    [UnivGate.Z, np.pi],
    [UnivGate.H, np.pi],
    [UnivGate.S, np.pi / 2],
    [UnivGate.T, np.pi / 4],
    [UnivGate.SD, np.pi / 2],
    [UnivGate.TD, np.pi / 4],
])
def test_rangle_std_gates(gate, expected):
    actual = rangle(gate)
    assert np.isclose(actual, expected), f'{actual} != {expected}'


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 100))
def test_rangle_negation_invariance(seed: int):
    """
    if operators u + v = 0, then their rotation angles are identical.
    """
    random.seed(seed)
    np.random.seed(seed)

    u = random_unitary(2)
    # print()
    # print(formatter.tostr(u))
    actual = rangle(u)
    expected = rangle(-u)
    # print(actual, expected)
    assert np.isclose(actual, expected)


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 100))
def test_rangle_random_su2(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    u = random_unitary(2)
    actual = rangle(u)
    assert 0 <= actual <= np.pi


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 100))
def test_rangle_random_unitary(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    u = random_unitary(2)
    angle = rangle(u)
    assert 0 <= angle <= np.pi


def test_rangle_slightly_gt_1():
    u = np.array([[(1.1588050842982336e-09 - 1j), (-7.003656213895806e-19 - 2.6411242922624396e-08j)], [(7.655884777871032e-18 - 2.8728853450477398e-08j), (-1.158805074151697e-09 - 1j)]])
    actual = rangle(u)
    assert np.isclose(actual, 0), f'{actual} != 0'


def test_rangle_eq_0():
    gate = -UnivGate.I.matrix
    actual = rangle(gate)
    assert np.isclose(actual, 0)


def test_rangle_eq_pi():
    gate = -UnivGate.X.matrix
    actual = rangle(gate)
    assert np.isclose(actual, np.pi)


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 100))
def test_rangle_sim(seed):
    random.seed(seed)
    np.random.seed(seed)

    angle = np.random.uniform(0, 2 * np.pi)
    # print(angle)

    uvec = np.random.standard_normal(3)
    u = rot(uvec, angle) * random_phase()

    vvec = np.random.standard_normal(3)
    v = rot(vvec, angle) * random_phase()
    assert np.isclose(rangle(u), rangle(v))


@pytest.mark.parametrize("u", [
    np.array([[1, 0]]),
    np.array([[1, 0, 1], [0, 1j, 3]]),
    np.array([[1, 0, 0, 1]]),
])
def test_dist_invalid_shapes(u):
    v = random_unitary(2)
    with pytest.raises(AssertionError) as e:
        dist(u, v)
    assert str(e.value) == "operators must have shape (2, 2)"


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 100))
def test_dist_zero(seed):
    random.seed(seed)
    np.random.seed(seed)

    u = random_unitary(2)
    d = dist(u, u)
    # print(f'distance: {d}')
    assert np.isclose(d, 0), f'{d} != 0'


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 10))
def test_dist_negation_zero_distance(seed: int):
    """
    Depending on the definition of distance function, the distance between two unitary matrices, u and -u, may be maximum or zero.
    Because our definition is phase agnostic, so the distance of two additive inverse is zero.
    """
    random.seed(seed)
    np.random.seed(seed)

    u = random_unitary(2)
    d = dist(u, -u)
    # print(f'distance: {d}')
    assert np.isclose(d, 0)


@pytest.mark.parametrize("u,v,angle", [
    [UnivGate.I, UnivGate.X, np.pi],
    [UnivGate.Z, UnivGate.H, np.pi / 2],
    [UnivGate.H, UnivGate.X, np.pi / 2],
    [UnivGate.I, UnivGate.Z, np.pi],
])
def test_dist_real_unitary(u, v, angle):
    actual = dist(u.matrix, v.matrix)
    expected = 2 * (1 - np.cos(angle / 2))
    assert np.isclose(actual, expected), f'{actual} != {expected}'


@pytest.mark.parametrize("u,v,angle", [
    [UnivGate.X, UnivGate.Y, np.pi],
    [UnivGate.I, UnivGate.Z, np.pi],
    [UnivGate.T, UnivGate.Y, np.pi],
    [UnivGate.T, UnivGate.TD, np.pi / 2],
    [UnivGate.S, UnivGate.TD, np.pi * 3 / 4],
    [UnivGate.S, UnivGate.H, np.pi * 2 / 3],
    [UnivGate.SD, UnivGate.Y, np.pi],
])
def test_dist_std_gates(u, v, angle):
    actual = dist(u.matrix, v.matrix)
    expected = 2 * (1 - np.cos(angle / 2))
    assert np.isclose(actual, expected), f'{actual} != {expected}'


@pytest.mark.parametrize("gate", list(UnivGate))
def test_dist_std_special_phase(gate):
    """
    There is a special phase 1j on standard UnivGate that demands special attention.
    Instead of being zero, the distance is 4.
    """
    gate = UnivGate.X
    u = gate.matrix
    assert dist(u, u) == 0
    assert dist(u, 1j * u) == 0
    assert dist(u, -1 * u) == 0
    assert dist(u, -1j * u) == 0


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 100))
def test_dist_phase_agnostic(seed):
    random.seed(seed)
    np.random.seed(seed)

    u = random_unitary(2)
    v = random_unitary(2)
    w = random_phase() * v
    assert np.isclose(dist(u, v), dist(u, w))


@pytest.mark.parametrize("gate,expected", [
    [UnivGate.I, np.array([0, 0, 1])],
    [UnivGate.X, np.array([1, 0, 0])],
    [UnivGate.Y, np.array([0, 1, 0])],
    [UnivGate.Z, np.array([0, 0, 1])],
    [UnivGate.H, np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2)])],
    [UnivGate.S, np.array([0, 0, 1])],
    [UnivGate.T, np.array([0, 0, 1])],
    [UnivGate.SD, np.array([0, 0, -1])],
    [UnivGate.TD, np.array([0, 0, -1])],
])
def test_raxis_std(gate, expected):
    u = gate.matrix
    # execute
    norm_vec = raxis(u)
    # verify
    assert np.isclose(np.linalg.norm(norm_vec), 1)
    angle = rangle(u)
    v = rot(norm_vec, angle) * gphase(u)
    assert np.allclose(v, u)
    assert np.allclose(norm_vec, expected), f'{norm_vec} != {expected}'


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 100))
def test_raxis_random_su2(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    angle = np.random.uniform(0, np.pi)
    vec = np.random.standard_normal(3)
    u = rot(vec, angle)

    # execute
    norm_vec = raxis(u)

    # verify
    assert np.isclose(np.linalg.norm(norm_vec), 1)
    expected = vec / np.linalg.norm(vec)
    assert np.allclose(norm_vec, expected), f'{norm_vec} != {expected}'


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 100))
def test_raxis_recover_unitary(seed):
    random.seed(seed)
    np.random.seed(seed)

    angle = np.random.uniform(0, np.pi)
    expected = np.random.standard_normal(3)
    u = rot(expected, angle) * random_phase()
    # execute
    norm_vec = raxis(u)
    # verify
    assert allprop(norm_vec, expected), f'{norm_vec} != {expected}'


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 100))
def test_raxis_rot_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)

    u = random_unitary(2)
    # execute
    angle = rangle(u)
    norm_vec = raxis(u)

    # verify
    assert np.isclose(np.linalg.norm(norm_vec), 1)
    u_recovered = rot(norm_vec, angle) * gphase(u)
    assert np.allclose(u_recovered, u)


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 100))
def test_raxis_complementary_angle(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    angle = np.random.uniform(0, 2 * np.pi)
    expected = np.random.standard_normal(3)
    # execute
    u = rot(expected, angle)
    v = rot(-expected, - angle)
    # verify
    assert np.allclose(u, v), f'\n{u} != \n{v}'


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 100))
def test_rot_verify_angle(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    angle = np.random.uniform(0, 2 * np.pi)
    uvec = np.random.standard_normal(3)
    vvec = np.random.standard_normal(3)
    u = rot(uvec, angle)
    v = rot(vvec, angle)
    assert np.isclose(rangle(u), rangle(v))


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 100))
def test_eigen_decompose_recoverable(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    u = random_unitary(2)
    phase, eigenval, eigenvec = eigen_decompose(u)
    # print(formatter.tostr(eigenval))
    assert eigenval.shape == (2,)
    actual = eigenvec @ np.diag(eigenval) @ herm(eigenvec) * phase
    assert np.allclose(actual, u)


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 1))
def test_eigen_decompose_tsim_verify(seed: int):
    seed = 936546
    random.seed(seed)
    np.random.seed(seed)

    angle = np.random.uniform(0, 2 * np.pi)
    uvec = np.random.standard_normal(3)
    vvec = np.random.standard_normal(3)
    u = rot(uvec, angle)
    v = rot(vvec, angle)
    uphase, uval, uvec = eigen_decompose(u)
    vphase, vval, vvec = eigen_decompose(v)
    assert np.allclose(uval, vval) or np.allclose(uval, vval[::-1])
    assert np.allclose(np.diag(uval) * uphase, herm(uvec) @ u @ uvec)
    assert np.allclose(np.diag(vval) * vphase, herm(vvec) @ v @ vvec)


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 100))
def test_tsim_similarity_su2(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    angle = np.random.uniform(0, 2 * np.pi)
    uvec = np.random.standard_normal(3)
    vvec = np.random.standard_normal(3)
    u = rot(uvec, angle)
    v = rot(vvec, angle)
    # print('v')
    # print(formatter.tostr(v))

    # execute
    t = tsim(u, v)

    actual = t @ u @ herm(t)
    # print('actual')
    # print(formatter.tostr(actual))
    assert np.allclose(actual, v)


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 100))
def test_tsim_similarity_unitary(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    angle = np.random.uniform(0, 2 * np.pi)
    uvec = np.random.standard_normal(3)
    vvec = np.random.standard_normal(3)
    uphase = random_phase()
    u = rot(uvec, angle) * uphase
    vphase = random_phase()
    v = rot(vvec, angle) * vphase
    # print('v')
    # print(formatter.tostr(v))

    # execute
    t = tsim(u, v)

    actual = t @ u @ herm(t) * vphase / uphase
    # print('actual')
    # print(formatter.tostr(actual))
    assert np.allclose(actual, v)


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 100))
def test_tsim_random_unitary_random_unitary(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    u = random_unitary(2)
    p = random_unitary(2)
    v = p @ u @ herm(p)

    # execute
    q = tsim(u, v)

    actual = q @ u @ herm(q)
    # print('actual')
    # print(formatter.tostr(actual))
    assert np.allclose(actual, v)


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 100))
def test_tsim_nonuniqueness(seed):
    random.seed(seed)
    np.random.seed(seed)

    """
    Similarity transformation are not unique. p @ u @ p† = q @ u @ q† does not necessarily mean p = q.
    In fact, suppose u is a rotation about an axis n. For any rotation, r, about the same axis, let q = p @ r.
    Then p @ u @ p† = q @ u @ q†.
    """

    u = random_unitary(2)
    p = random_unitary(2)
    # print('p')
    # print(formatter.tostr(p))
    v = p @ u @ herm(p)

    # execute
    q = tsim(u, v)
    # print('q')
    # print(formatter.tostr(q))

    # verify
    actual = q @ u @ herm(q)
    assert np.allclose(actual, v)
    # sim transformation is not unique
    assert np.allclose(p @ u @ herm(p), q @ u @ herm(q))

    assert not np.allclose(p, q)


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 100))
def test_gc_decompose_positive_trace(seed):
    """
    Positively-traced unitary has negative phase.
    """
    random.seed(seed)
    np.random.seed(seed)

    expected = random_su2()
    expected = expected / gphase(expected)
    assert np.isclose(np.linalg.det(expected), 1)
    assert np.isclose(gphase(expected), 1)
    assert np.trace(expected) > 0

    # print('expected')
    # print(formatter.tostr(expected))
    v, w = gc_decompose(expected)
    actual = -v @ w @ herm(v) @ herm(w)
    # print('actual')
    # print(formatter.tostr(actual))
    assert np.allclose(actual, expected)


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 10))
def test_gc_decompose_negative_trace(seed):
    """
    Negatively-traced unitary has positive phase.
    """
    random.seed(seed)
    np.random.seed(seed)

    expected = random_su2()
    expected = -expected / gphase(expected)
    assert np.isclose(np.linalg.det(expected), 1)
    assert np.isclose(gphase(expected), -1)
    assert np.trace(expected) < 0

    # print('expected')
    # print(formatter.tostr(expected))
    v, w = gc_decompose(expected)
    actual = v @ w @ herm(v) @ herm(w)
    # print('actual')
    # print(formatter.tostr(actual))
    assert np.allclose(actual, expected)


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 100))
def test_gc_decompose_random_unitary(seed):
    """
    Positively-traced unitary has negative sign.
    """
    random.seed(seed)
    np.random.seed(seed)

    expected = random_unitary(2)
    # print('expected')
    # print(formatter.tostr(expected))
    v, w = gc_decompose(expected)
    phase = gphase(expected)
    sign = phase if np.trace(expected / phase) < 0 else -phase
    actual = sign * v @ w @ herm(v) @ herm(w)
    # print('actual')
    # print(formatter.tostr(actual))
    assert np.allclose(actual, expected)


@pytest.mark.parametrize("x,y,expected", [
    # [-1j, 1j, 5 / 4 * np.pi],  # invalid input
    [0, 0, 0],
    [0, 1, np.pi / 2],
    [1, 0, 0],
    [1, 1, np.pi / 4],
    [-1, 1, 3 / 4 * np.pi],
    [1, -1, -np.pi / 4],
    [-1, -1, -3 / 4 * np.pi],
    [-10, 10, 3 / 4 * np.pi],
])
def test_tan2(x, y, expected):
    angle = np.arctan2(y, x)
    assert angle == expected


@pytest.mark.parametrize("gate,expected", [
    [UnivGate.I, (0, 0, 0)],
    [UnivGate.X, (np.pi / 2, 0, np.pi)],
    [UnivGate.Y, (np.pi / 2, np.pi / 2, np.pi)],
    [UnivGate.Z, (0, 0, np.pi)],
    [UnivGate.H, (np.pi / 4, 0, np.pi)],
    [UnivGate.S, (0, 0, np.pi / 2)],
    [UnivGate.T, (0, 0, np.pi / 4)],
    [UnivGate.SD, (np.pi, 0, np.pi / 2)],
    [UnivGate.TD, (np.pi, 0, np.pi / 4)],
])
def test_vec_std(gate, expected):
    angle_vec = vec(gate.matrix)
    assert np.allclose(angle_vec, expected, rtol=1.e-4, atol=1.e-7), f"{angle_vec} != {expected}"


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 100))
def test_vec_negation(seed):
    random.seed(seed)
    np.random.seed(seed)

    u = random_unitary(2)
    v = -u

    # execute
    theta1, phi1, alpha1 = vec(u)
    theta2, phi2, alpha2 = vec(v)

    # verify
    assert theta1 == theta2
    assert phi1 == phi2
    assert alpha1 == alpha2


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 100))
def test_vec_hermite(seed):
    random.seed(seed)
    np.random.seed(seed)

    u = random_unitary(2)
    v = herm(u)

    # execute
    theta1, phi1, alpha1 = vec(u)
    theta2, phi2, alpha2 = vec(v)

    # verify axis are opposite to one another
    assert np.isclose(alpha1, alpha2)
    assert np.isclose(theta1 + theta2, np.pi)
    assert np.isclose(np.abs(phi1 - phi2), np.pi), f"{phi1 - phi2} != π"


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 100))
def test_vec_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)

    angle = np.random.uniform(0, np.pi)
    uvec = np.random.standard_normal(3)
    u = rot(uvec, angle) * random_phase()

    # execute
    theta, phi, alpha = vec(u)
    # verify
    assert np.isclose(alpha, angle)
    assert np.isclose(theta, np.arccos(uvec[2] / np.linalg.norm(uvec)))
    assert np.isclose(phi, np.arctan2(uvec[1], uvec[0]))


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 100))
def test_vec_random(seed):
    random.seed(seed)
    np.random.seed(seed)

    u = random_unitary(2)
    # execute
    actual = vec(u)

    # verify
    assert np.isclose(actual[2], rangle(u))
    x, y, z = raxis(u)
    assert np.isclose(actual[0], np.arccos(z))
    assert np.isclose(actual[1], np.arctan2(y, x))


@pytest.mark.skip(reason="These tests are not finished yet)")
@pytest.mark.parametrize('x,y,expected', [
    [[1e-1, 1e-1 - np.pi, 1e-1], [np.pi, np.pi, np.pi], 3e-1],
    [[1, 1, 1], [1, 1, 1], 0],
    [[np.pi - 1e-1, np.pi - 1e-1, 0], [0, -np.pi, np.pi - 1e-1], 3e-1],
])
def test_mod_dist(x, y, expected):
    d = mod_dist(np.array(x), np.array(y))
    assert np.isclose(d, expected), f'{d} != {expected}'


ANGLE_RANGES = np.array([[0, np.pi], [-np.pi, np.pi], [0, np.pi]])


def test_boundary_vec():
    # Specify resolution (number of steps per axis)
    res = np.array([3, 3, 3])  # nx, ny, nz

    # Generate linspaces per axis (excluding upper bound)
    axes = [
        np.linspace(start, stop, num=n, endpoint=True)
        for (start, stop), n in zip(ANGLE_RANGES, res)
    ]
    # Generate grid
    mesh = np.meshgrid(*axes, indexing='ij')  # shape: (3 arrays of shape [nx, ny, nz])
    grid = np.stack(mesh, axis=-1).reshape(-1, 3)  # (N, 3)
    print(grid)
    us = [rota(*v) for v in grid]
    # print(us)
    nvecs = [vec(u) for u in us]
    comparison = np.hstack([grid, nvecs])
    idx = np.argsort(comparison[:, ::-1])
    print()
    print(comparison[idx])


def test_verify_eq():
    scoord = np.array([np.pi / 2, 0, np.pi])
    r = rota(*scoord)
    nvec = vec(r)
    print()
    print(nvec)
