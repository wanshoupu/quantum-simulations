import random

import numpy as np
import pytest

from quompiler.construct.qspace import QSpace, Qubit


def test_qspace_init_random():
    dim = 100
    for _ in range(10):
        qids = np.random.choice(dim, size=10, replace=False)
        qs = QSpace(qids)
        assert qs
        sorting = np.argsort(qids).tolist()
        # print(sorting)
        assert sorting == qs.sorting


def test_qspace_iter():
    pass


@pytest.mark.parametrize("qids,n,expected", [
    [[10, 0, 7, 1, 8], 0b10101, 0b11100],
])
def test_qspace_map(qids, n, expected):
    qs = QSpace(qids)
    actual = qs.map(n)
    assert actual == expected


def test_qspace_map_random():
    dim = 100
    for _ in range(10):
        k = random.randint(1, 10)
        qids = np.random.choice(dim, size=k, replace=False)
        qs = QSpace(qids)
        indexes = list(range(1 << k))
        actual = [qs.map(i) for i in indexes]
        # print(f'mapped={actual}')
        assert sorted(actual) == indexes


@pytest.mark.parametrize("qids,expected", [
    [[10, 0, 7, 1, 8], False],
    [[0, 1, 7, 8, 10], True],
])
def test_qspace_is_sorted(qids, expected):
    qs = QSpace(qids)
    assert qs.is_sorted() == expected


def test_qspace_map_all_random():
    dim = 100
    for _ in range(10):
        k = random.randint(1, 10)
        qids = np.random.choice(dim, size=k, replace=False)
        qs = QSpace(qids)
        size = random.randint(1, 1 << k)
        indexes = np.random.choice(1 << k, size=size, replace=False)
        expected = [qs.map(i) for i in indexes]
        actual = qs.map_all(indexes)
        assert actual == expected


def test_qubit_init_invalid():
    q0 = Qubit(0)
    assert q0 is not None
    with pytest.raises(AssertionError):
        Qubit(-50)


def test_qubit_init_comparison():
    assert Qubit(1) == Qubit(1)
    assert Qubit(1) < Qubit(2)
    assert Qubit(51) >= Qubit(0)


def test_qubits_sorting():
    n = 10
    qubits = [Qubit(random.randint(0, 10)) for _ in range(n)]
    qubits.sort()
    assert all(qubits[i - 1] <= qubits[i] for i in range(1, n))
    assert all(qubits[i] >= qubits[i - 1] for i in range(1, n))

    qset = set(qubits)
    assert len(qset) < len(qubits)
