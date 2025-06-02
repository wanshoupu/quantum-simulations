import random
from functools import reduce

import numpy as np
import pytest

from quompiler.construct.bytecode import BytecodeIter
from quompiler.construct.solovay import SKDecomposer
from quompiler.construct.types import UnivGate
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.group_su2 import dist
from quompiler.utils.mgen import random_su2, random_gate_seq

formatter = MatrixFormatter(precision=3)


@pytest.mark.parametrize('rtol, atol, depth', [
    [1.e3, 1.e5, 0],
    [1.e-3, 1.e-5, 23],
    [1.e-5, 1.e-8, 36],
    [1, 1, 0],
])
def test_init_verify_depth(rtol, atol, depth):
    sk = SKDecomposer(rtol=rtol, atol=atol)
    assert sk.depth == 0


@pytest.mark.parametrize('gate', list(UnivGate))
def test_approx_std(gate):
    sk = SKDecomposer(rtol=1.e-2, atol=1.e-3, lookup_error=.2)
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
    sk = SKDecomposer(rtol=1.e-1, atol=1.e-2, lookup_error=.3)
    original = np.array(gate1 @ gate2)
    gates = sk.approx(original)
    approx = np.array(reduce(lambda x, y: x @ y, gates))
    error = dist(original, approx)
    print(f'\nGate {gate1 @ gate2} decomposed into {len(gates)} gates, with error {error}.')
    print(f'approx: \n{formatter.tostr(approx)}')
    assert np.isclose(error, 0)


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 10))
def test_approx_random(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    sk = SKDecomposer(lookup_error=.3)
    original = random_su2()
    # print(f'original: \n{formatter.tostr(original)}')
    gates = sk.approx(original)
    # print(gates)
    approx = np.array(reduce(lambda x, y: x @ y, gates))
    error = dist(original, approx)
    print(f'\nLookup_error={sk.lookup_error}: Gate {original} decomposed with error {formatter.nformat(error)}, into gates: {gates}')
    print(f'approx:\n{formatter.tostr(approx)}')
    assert np.isclose(error, 0, rtol=sk.rtol, atol=sk.atol)


@pytest.mark.skip(reason="Temporarily disabling this test for its long runtime.")
@pytest.mark.parametrize("atol", [1, 0.1, 1e-2])
def test_approx_scalability(atol: float):
    rtol = 1.e8

    sk = SKDecomposer(rtol=rtol, atol=atol, lookup_error=.2)
    original = random_su2()
    # print(f'original: \n{formatter.tostr(original)}')
    gates = sk.approx(original)
    # print(gates)
    approx = np.array(reduce(lambda x, y: x @ y, gates))
    error = dist(original, approx)
    met = 'met' if np.isclose(error, 0, atol=atol, rtol=rtol) else 'unmet'
    print()
    print(f'\natol={atol}: Gate {original} decomposed into {len(gates)} gates, with error {formatter.nformat(error)}. requirement {met}.')


@pytest.mark.skip(reason="These tests are for manual run only (comment out this line to run and do not commit changes)")
def test_approx_depth_plus_one():
    """
    This is to debug why sk error is diverging.
    """
    seed = 428
    random.seed(seed)
    np.random.seed(seed)

    sk = SKDecomposer(rtol=1.e-1, atol=1.e-2, lookup_error=.15)
    seq = random_gate_seq(sk.depth + 15)
    original = reduce(lambda x, y: x @ y, seq)
    while True:
        direct_lookup_node, _ = sk.su2net.lookup(original)
        direct_lookup_gates = [n.data for n in BytecodeIter(direct_lookup_node) if n.is_leaf()]
        direct_lookup_approx = np.array(reduce(lambda x, y: x @ y, direct_lookup_gates))
        direct_lookup_error = dist(original, direct_lookup_approx)
        if 1.e-3 < direct_lookup_error:
            break
        seq = random_gate_seq(sk.depth + 15)
        original = reduce(lambda x, y: x @ y, seq)

    one_iter_node = sk._sk_decompose(original, 1)
    one_iter_gates = [n.data for n in BytecodeIter(one_iter_node) if n.is_leaf()]
    one_iter_approx = np.array(reduce(lambda x, y: x @ y, one_iter_gates))
    one_iter_error = dist(original, one_iter_approx)
    assert np.allclose(one_iter_node.data, one_iter_approx)

    print(
        f'\nDirect lookup decomposed {seq}=\n{formatter.tostr(original, indent=4)}\ninto {direct_lookup_gates} gates=\n{formatter.tostr(direct_lookup_approx, indent=4)}\nwith error {direct_lookup_error}.')
    print(f'\nOne iter decomposed {seq}=\n{formatter.tostr(original, indent=4)}\ninto {one_iter_gates} gates=\n{formatter.tostr(one_iter_approx, indent=4)}\nwith error {one_iter_error}.')
