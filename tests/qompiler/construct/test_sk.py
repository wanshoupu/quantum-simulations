from functools import reduce

import numpy as np
import pytest

from quompiler.construct.solovay import SKDecomposer
from quompiler.construct.types import UnivGate
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.group_su2 import dist
from quompiler.utils.mfun import herm
from quompiler.utils.mgen import random_su2

formatter = MatrixFormatter(precision=2)


def test_init_verify_depth():
    rtol = 1.e-3
    atol = 1.e-5
    sk = SKDecomposer(rtol=rtol, atol=atol)
    assert sk.depth == 11


def test_approx():
    rtol = 1.e-2
    atol = 1.e-3
    sk = SKDecomposer(rtol=rtol, atol=atol)
    original = random_su2()
    # print(f'original: \n{formatter.tostr(original)}')
    gates = sk.approx(original)
    assert np.log(len(gates)).astype(int) == 10
    approx = reduce(lambda x, y: x @ y, gates)
    error = dist(original, approx)
    # print(f'\nGate {original} decomposed into {len(gates)} gates, with error {error}.')
    # print(f'approx: \n{formatter.tostr(approx)}')


@pytest.mark.parametrize('gate', list(UnivGate))
def test_approx_std(gate):
    rtol = 1.e-2
    atol = 1.e-3
    sk = SKDecomposer(rtol=rtol, atol=atol)
    original = np.array(gate)
    gates = sk.approx(original)
    approx = reduce(lambda x, y: x @ y, gates)
    error = dist(original, approx)
    print(f'\nGate {gate} decomposed into {len(gates)} gates, with error {error}.')
    print(f'approx: \n{formatter.tostr(approx)}')
    # assert np.isclose(error, 0)
