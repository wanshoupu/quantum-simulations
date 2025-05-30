from functools import reduce

import numpy as np
import pytest

from quompiler.construct.solovay import SKDecomposer
from quompiler.construct.types import UnivGate
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.group_su2 import dist
from quompiler.utils.mgen import random_su2, cyclic_matrix

formatter = MatrixFormatter(precision=2)


def test_init_verify_depth():
    rtol = 1.e-3
    atol = 1.e-5
    sk = SKDecomposer(rtol=rtol, atol=atol)
    assert sk.depth == 11


@pytest.mark.parametrize('gate', list(UnivGate))
def test_approx_std(gate):
    rtol = 1.e-2
    atol = 1.e-3
    sk = SKDecomposer(rtol=rtol, atol=atol)
    original = np.array(gate)
    gates = sk.approx(original)
    approx = np.array(reduce(lambda x, y: x @ y, gates))
    error = dist(original, approx)
    print(f'\nGate {gate} decomposed into {len(gates)} gates, with error {error}.')
    print(f'approx: \n{formatter.tostr(approx)}')
    assert np.isclose(error, 0)


@pytest.mark.parametrize("gate1,gate2", [
    [UnivGate.X, UnivGate.Y],
    [UnivGate.Y, UnivGate.Z],
    [UnivGate.H, UnivGate.Y],
    [UnivGate.S, UnivGate.Y],
    [UnivGate.T, UnivGate.H],
    [UnivGate.SD, UnivGate.X],
    [UnivGate.TD, UnivGate.H],
])
def test_approx_std(gate1, gate2):
    rtol = 1.e-2
    atol = 1.e-3
    sk = SKDecomposer(rtol=rtol, atol=atol)
    original = np.array(gate1 @ gate2)
    gates = sk.approx(original)
    approx = np.array(reduce(lambda x, y: x @ y, gates))
    error = dist(original, approx)
    print(f'\nGate {gate1 @ gate2} decomposed into {len(gates)} gates, with error {error}.')
    print(f'approx: \n{formatter.tostr(approx)}')
    assert np.isclose(error, 0)


@pytest.mark.skip(reason="Temporarily disabling this test for its long runtime.")
def test_approx_random():
    rtol = 1.e-3
    atol = 1.e-4
    sk = SKDecomposer(rtol=rtol, atol=atol)
    for _ in range(3):
        print(f'Test round {_}')
        original = random_su2()
        # print(f'original: \n{formatter.tostr(original)}')
        gates = sk.approx(original)
        assert np.log(len(gates)).astype(int) == 10
        approx = np.array(reduce(lambda x, y: x @ y, gates))
        error = dist(original, approx)
        print(f'\nGate {original} decomposed into {len(gates)} gates, with error {error}.')
        print(f'approx: \n{formatter.tostr(approx)}')
