import random

import numpy as np
import pytest

from quompiler.construct.qspace import Qubit
from quompiler.utils.shuffle import Shuffler


def test_shuffle_init():
    n = 10
    qubits = [Qubit(i) for i in range(n)]
    random.shuffle(qubits)
    print(qubits)
    sh = Shuffler(qubits)
    indexes = sh.map_all(range(n))
    print(indexes)


@pytest.mark.parametrize("origin,permute,expected", [
    [[10, 0, 7, 1, 8], [10, 0, 7, 1, 8], [0, 1, 2, 3, 4]],
    [[10, 0, 7, 1, 8], [8, 10, 0, 7, 1], [1, 2, 3, 4, 0]],
    [[10, 0, 7, 1, 8], [8, 7, 0, 10, 1], [3, 2, 1, 4, 0]],
    [[0, 10, 1, 7, 8], [8, 7, 0, 10, 1], [2, 3, 4, 1, 0]],
])
def test_create_from_permute(origin, permute, expected):
    sh = Shuffler.from_permute(origin, permute)
    assert sh.sorting == expected


def test_create_from_permute_random():
    for _ in range(100):
        n = random.randint(1, 11)
        original = random.sample(range(n), n)
        shuffled = random.sample(range(n), n)

        # execute
        sh = Shuffler.from_permute(original, shuffled)

        # verify
        actual = [shuffled[i] for i in sh.sorting]
        assert actual == original


def test_create_from_init_equivalence():
    for _ in range(100):
        n = random.randint(1, 100)
        shuffled = random.sample(range(n), n)

        # execute
        actual = Shuffler.from_permute(range(n), shuffled).sorting
        expected = Shuffler(shuffled).sorting

        # verify
        assert actual == expected


@pytest.mark.parametrize("qids,n,expected", [
    [[10, 0, 7, 1, 8], 0b10101, 0b11100],
])
def test_map_indexes(qids, n, expected):
    qs = Shuffler(qids)
    actual = qs.map(n)
    assert actual == expected


def test_map_verify():
    indexes = [10, 0, 7, 1, 8]
    sh = Shuffler(indexes)
    for _ in range(100):
        n = random.randrange(0b100000)
        actual = sh.map(n)
        expected = map_equiv(sh, n)
        assert actual == expected


def map_equiv(shf, n):
    # In little endian
    bits = bin(n)[2:].zfill(5)[::-1]
    shuffled_bits = shf.shuffle(bits)[::-1]
    return int(''.join(shuffled_bits), base=2)


def test_shuffle():
    indexes = [10, 0, 7, 1, 8]
    qs = Shuffler(indexes)
    assert qs.sorting == [1, 3, 2, 4, 0]
    items = list('abcde')
    shuffled = qs.shuffle(items)
    print(shuffled)
    assert shuffled == list('bdcea')


def test_map_random():
    dim = 100
    for _ in range(10):
        k = random.randint(1, 10)
        indexes = np.random.choice(dim, size=k, replace=False)
        qs = Shuffler(indexes)
        indexes = list(range(1 << k))
        actual = [qs.map(i) for i in indexes]
        # print(f'mapped={actual}')
        assert sorted(actual) == indexes


def test_map_all_4():
    qs = Shuffler([3, 0])
    indexes = list(range(4))
    mapped = qs.map_all(indexes)
    print(mapped)


def test_map_all_random():
    dim = 100
    for _ in range(10):
        k = random.randint(1, 10)
        indexes = np.random.choice(dim, size=k, replace=False)
        qs = Shuffler(indexes)
        size = random.randint(1, 1 << k)
        indexes = np.random.choice(1 << k, size=size, replace=False)
        expected = [qs.map(i) for i in indexes]
        actual = qs.map_all(indexes)
        assert actual == expected


def test_create_from_permute_invalid():
    origin = [1, 2, 3]
    permute = [0, 1]
    with pytest.raises(AssertionError):
        Shuffler.from_permute(origin, permute)


def test_map_all_idempotency():
    n = 5
    seq = list(range(n))
    random.shuffle(seq)
    sh = Shuffler(seq)
    actual = tuple(range(1 << n))
    lookup = set()
    for _ in range(1 << n):
        if actual not in lookup:
            lookup.add(actual)
        else:
            actual = tuple(sh.map_all(actual))
    assert len(lookup) == 3
