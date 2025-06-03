import random
from functools import reduce

import numpy as np
import pytest

from quompiler.construct.su2net import SU2Net
from quompiler.construct.types import SU2NetType, UnivGate
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.group_su2 import dist
from quompiler.utils.mfun import herm
from quompiler.utils.mgen import random_su2, random_unitary

formatter = MatrixFormatter(precision=2)


class SU2NetTestTemplate:
    su2net: SU2Net = None

    def test_init_lazy(self):
        # tree not constructed until first lookup
        assert not self.su2net.constructed

    def test_lookup_identity(self):
        gate = UnivGate.I
        matrix = np.array(gate)
        node, lookup_error = self.su2net.lookup(matrix)
        seq = [n.data for n in node.children]
        v = reduce(lambda a, b: a @ b, seq, np.eye(2))
        assert np.allclose(np.array(v), node.data)  # self-consistent
        error = dist(np.array(v), np.array(matrix))
        assert error == lookup_error == 0

    @pytest.mark.parametrize('gate', list(UnivGate))
    def test_lookup_std(self, gate):
        matrix = np.array(gate)
        node, lookup_error = self.su2net.lookup(matrix)
        seq = [n.data for n in node.children]
        v = reduce(lambda a, b: a @ b, seq, np.eye(2))
        assert np.allclose(np.array(v), node.data)  # self-consistent
        error = dist(np.array(v), np.array(matrix))
        assert np.isclose(error, lookup_error)
        assert np.isclose(error, 0)

    @pytest.mark.parametrize("seed", random.sample(range(1 << 20), 10))
    def test_lookup_random_su2(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)

        u = random_su2()
        node, lookup_error = self.su2net.lookup(u)
        leaves = [n for n in node.children]
        assert all(l.is_leaf() for l in leaves)
        coms = [l.data for l in leaves]
        v = reduce(lambda a, b: a @ b, coms, np.eye(2))
        assert np.allclose(np.array(v), node.data)  # self-consistent
        error = dist(np.array(v), np.array(u))
        # print(f'\n{formatter.tostr(u)}\nlookup error: {lookup_error}, dist: {error}')
        assert error < self.su2net.error

    @pytest.mark.parametrize("seed", random.sample(range(1 << 20), 10))
    def test_lookup_random_unitary_verify_error_margin(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)

        u = random_unitary(2)
        node, lookup_error = self.su2net.lookup(u)
        coms = [n.data for n in node.children]
        v = reduce(lambda a, b: a @ b, coms, np.eye(2))
        assert np.allclose(np.array(v), node.data)  # self-consistent
        error = dist(np.array(v), np.array(u))
        # print(f'\n{formatter.tostr(u)}\nlookup error: {lookup_error}, dist error: {error}')
        # print(f'gphase error: {gphase(v @ herm(u))}')
        assert error < self.su2net.error

    @pytest.mark.parametrize("seed", random.sample(range(1 << 20), 10))
    def test_lookup_random_unitary_verify_unitarity(self, seed: int):
        """
        Verify that the result of lookup is still unitary with unit absolute det
        """
        random.seed(seed)
        np.random.seed(seed)

        u = random_unitary(2)
        # execute
        node, lookup_error = self.su2net.lookup(u)
        # verify
        det = np.abs(np.linalg.det(node.data))
        # print(f'det: {det}')
        assert np.isclose(det, 1)
        unitarity = np.sqrt(np.sum((node.data @ herm(node.data) - np.eye(2)) ** 2))
        # print(f'unitarity error: {unitarity}')
        assert np.isclose(unitarity, 0)


class TestAutoNN(SU2NetTestTemplate):
    su2net: SU2Net = SU2Net(.4, SU2NetType.AutoNN)


@pytest.mark.skip(reason="These tests are failing sporadically)")
class TestBruteNN(SU2NetTestTemplate):
    su2net: SU2Net = SU2Net(.4, SU2NetType.BruteNN)


class TestBallTreeNN(SU2NetTestTemplate):
    su2net: SU2Net = SU2Net(.4, SU2NetType.BallTreeNN)


class TestKDTreeNN(SU2NetTestTemplate):
    su2net: SU2Net = SU2Net(.4, SU2NetType.KDTreeNN)
