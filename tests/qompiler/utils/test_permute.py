import random

import numpy as np
import pytest

from quompiler.construct.qspace import Qubit
from quompiler.utils.permute import Permuter


def test_permute_init():
    n = 10
    qubits = [Qubit(i) for i in range(n)]
    random.shuffle(qubits)
    print(qubits)
    perm = Permuter(qubits)
    indexes = perm.bitsortall(range(n))
    print(indexes)


@pytest.mark.parametrize("seq,_sort,_perm", [
    [[10, 0, 7, 1, 8], [1, 3, 2, 4, 0], [4, 0, 2, 1, 3]],
    [[0, 7, 10, 1, 8], [0, 3, 1, 4, 2], [0, 2, 4, 1, 3]],
])
def test_sorting_order_permute_order(seq, _sort, _perm):
    perm = Permuter(seq)
    assert perm._sort == _sort
    assert perm._perm == _perm


@pytest.mark.parametrize("origin,permute,expected", [
    [[10, 0, 7, 1, 8], [10, 0, 7, 1, 8], [0, 1, 2, 3, 4]],
    [[10, 0, 7, 1, 8], [8, 10, 0, 7, 1], [1, 2, 3, 4, 0]],
    [[10, 0, 7, 1, 8], [8, 7, 0, 10, 1], [3, 2, 1, 4, 0]],
    [[0, 10, 1, 7, 8], [8, 7, 0, 10, 1], [2, 3, 4, 1, 0]],
])
def test_create_from_permute(origin, permute, expected):
    perm = Permuter.from_permute(origin, permute)
    assert perm._sort == expected


def test_create_from_permute_random():
    for _ in range(100):
        n = random.randint(1, 11)
        original = random.sample(range(n), n)
        permuted = random.sample(range(n), n)

        # execute
        perm = Permuter.from_permute(original, permuted)

        # verify
        actual = perm.sort(permuted)
        assert actual == original

        actual2 = perm.permute(original)
        assert actual2 == permuted


def test_create_from_permute_invariance():
    for _ in range(100):
        n = random.randint(1, 11)
        original = random.sample(range(n), n)
        permuted = random.sample(range(n), n)

        # execute
        perm = Permuter.from_permute(original, permuted)

        # verify
        expected = random.sample(range(n), n)
        permute_sort = perm.permute(perm.sort(expected))
        assert permute_sort == expected

        sort_permute = perm.sort(perm.permute(expected))
        assert sort_permute == expected


def test_create_from_init_equivalence():
    for _ in range(100):
        n = random.randint(1, 100)
        permuted = random.sample(range(n), n)

        # execute
        actual = Permuter.from_permute(range(n), permuted)._sort
        expected = Permuter(permuted)._sort

        # verify
        assert actual == expected


@pytest.mark.parametrize("seq,original,expected", [
    [[0, 7, 10, 1, 8], list('abcde'), list('adbec')],
])
def test_sort_bits(seq, original, expected):
    qs = Permuter(seq)
    actual = qs.sort(original)
    print()
    print(actual)
    assert actual == expected, f'{actual} != {expected}'


@pytest.mark.parametrize("seq,original,expected", [
    [[0, 7, 10, 1, 8], list('abcde'), list('acebd')],
])
def test_permute_bits(seq, original, expected):
    qs = Permuter(seq)
    actual = qs.permute(original)
    print()
    print(actual)
    assert actual == expected, f'{actual} != {expected}'


@pytest.mark.parametrize("seq,n,expected", [
    [[0, 1, 3, 2, 4, 5, 6, 7], 0b11101101, 0b11011101],
    [[0, 2, 1, 3, 4, 5, 6, 7], 0b11101101, 0b11101101],
    [[0, 1, 2, 3, 4, 5, 7, 6], 0b11101101, 0b11101110],
    [[0, 1, 2, 3, 4, 6, 5, 7], 0b11101101, 0b11101011],
    [[0, 1, 2, 3, 6, 4, 5, 7], 0b11101101, 0b11100111],
    [[0, 7, 1, 8, 10], 0b10101, 0b11001],
    [[1, 0, 7, 8, 10], 0b10101, 0b1101],
    [[0, 7, 1, 10, 8], 0b10101, 0b11010],
    [[0, 7, 10, 1, 8], 0b10101, 0b11100],
    [[10, 0, 7, 1, 8], 0b10101, 0b11100],
])
def test_bitpermute(seq, n, expected):
    qs = Permuter(seq)
    actual = qs.bitpermute(n)
    print()
    print(bin(actual))
    assert actual == expected, f'{bin(actual)} != {bin(expected)}'


def test_bitsort_verify():
    indexes = [10, 0, 7, 1, 8]
    perm = Permuter(indexes)
    for _ in range(100):
        n = random.randrange(0b100000)
        actual = perm.bitsort(n)
        expected = bitsort_equiv(perm, n)
        assert actual == expected


def bitsort_equiv(shf, n):
    # In little endian
    bits = bin(n)[2:].zfill(5)
    permuted_bits = shf.sort(bits)
    return int(''.join(permuted_bits), base=2)


def test_sort_and_bits():
    indexes = [10, 0, 7, 1, 8]
    qs = Permuter(indexes)
    assert qs._sort == [1, 3, 2, 4, 0]
    items = list('abcde')
    permuted = qs.sort(items)
    print(permuted)
    assert permuted == list('bdcea')


def test_map_random():
    dim = 100
    for _ in range(10):
        k = random.randint(1, 10)
        indexes = np.random.choice(dim, size=k, replace=False)
        qs = Permuter(indexes)
        indexes = list(range(1 << k))
        actual = [qs.bitsort(i) for i in indexes]
        # print(f'mapped={actual}')
        assert sorted(actual) == indexes


def test_map_all_4():
    qs = Permuter([3, 0])
    indexes = list(range(4))
    mapped = qs.bitsortall(indexes)
    print(mapped)


def test_map_all_random():
    dim = 100
    for _ in range(10):
        k = random.randint(1, 10)
        indexes = np.random.choice(dim, size=k, replace=False)
        qs = Permuter(indexes)
        size = random.randint(1, 1 << k)
        indexes = np.random.choice(1 << k, size=size, replace=False)
        expected = [qs.bitsort(i) for i in indexes]
        actual = qs.bitsortall(indexes)
        assert actual == expected


def test_create_from_permute_invalid():
    origin = [1, 2, 3]
    permute = [0, 1]
    with pytest.raises(AssertionError):
        Permuter.from_permute(origin, permute)


def test_map_all_idempotency():
    n = 5
    seq = list(range(n))
    random.shuffle(seq)
    perm = Permuter(seq)
    actual = tuple(range(1 << n))
    lookup = set()
    for _ in range(1 << n):
        if actual not in lookup:
            lookup.add(actual)
        else:
            actual = tuple(perm.bitsortall(actual))
    assert len(lookup) == 3
