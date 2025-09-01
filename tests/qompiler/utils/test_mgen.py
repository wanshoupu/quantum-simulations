from itertools import product
from random import random

import cirq

from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import *
import numpy as np

formatter = MatrixFormatter()


def test_2l():
    m2l = random_matrix_2l(10, 1, 6)
    # print(formatter.tostr(m2l))


def test_unitary():
    randu = random_unitary(2)
    # print(formatter.tostr(randu))
    identity = randu.T @ np.conj(randu)
    assert np.all(np.isclose(identity, np.eye(*identity.shape))), print(formatter.tostr(identity))


def test_permeye():
    for _ in range(10):
        n = random.randint(10, 16)
        a = random.randrange(n)
        b = random.randrange(n)
        xs = xindexes(n, a, b)
        pi = permeye(xs)
        if a == b:
            assert pi[a, a] == 1 == pi[b, b], f'diagonal {a},{b}\n{pi}'
        else:
            assert pi[a, b] == 1 == pi[b, a], f'off diagonal {a},{b}\n{pi}'
            assert pi[a, a] == 0 == pi[b, b], f'diagonal {a},{b}\n{pi}'


def test_cyclic():
    cm = cyclic_matrix(8, 2)
    # print(formatter.tostr(cm))


def test_xindexes():
    for _ in range(10):
        n = random.randint(10, 100)
        a = random.randrange(n)
        b = random.randrange(n)
        xs = xindexes(n, a, b)
        assert xs[a] == b and xs[b] == a


def test_random_indexes():
    for _ in range(10):
        n = random.randint(10, 100)
        size = random.randrange(1, n)
        indxs = random_indexes(n, size)
        assert len(indxs) == len(set(indxs)), "Indexes contain duplicates!"
        assert 0 <= min(indxs) and max(indxs) < n, "Indexes are out of boundaries!"


def test_random_control():
    for _ in range(10):
        n = random.randint(10, 100)
        size = random.randrange(n)
        control = random_control(n, size)
        assert len(control) == n, f'Length of control {len(control)} is not {n}'
        assert control.count(QType.TARGET) == size, f'Number of target bits, {control.count(QType.TARGET)}, does not equal to the expected {size}'


def test_gen_gate_seq_default_set():
    seq = random_gate_seq(15)
    expected = [UnivGate[l] for l in 'X,H,TD,SD,T,S,I,S,Y,H,I,S,T,SD,H,T'.split(',')]
    assert seq == expected


def test_gen_gate_seq_custom_candidates():
    cands = UnivGate.cliffordt()
    seq = random_gate_seq(15, cands)
    expected = [UnivGate[l] for l in 'X,H,X,S,I,S,I,X,H,X,TD,T,H,TD,SD,H'.split(',')]
    assert seq == expected


def test_qft_matrix():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.S(q0).controlled_by(q1),
        cirq.T(q0).controlled_by(q2),
        cirq.H(q1),
        cirq.S(q1).controlled_by(q2),
        cirq.H(q2),
        cirq.SWAP(q0, q2),
    )
    actual = circuit.unitary()
    # print(formatter.tostr(actual))
    expected = fft_matrix(3)
    # print(formatter.tostr(expected))
    assert np.allclose(actual, expected)
