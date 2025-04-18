from typing import Iterable

from quompiler.utils.format_matrix import MatrixFormatter
import random
import numpy as np

from quompiler.utils.gray import gray_code, control_bits, cogray_code

random.seed(5)
formatter = MatrixFormatter()


def test_gray_code():
    for _ in range(10):
        a = random.randint(10, 100)
        b = random.randint(10, 100)
        # print(f'{a}, {b}')
        blength = max(a.bit_length(), b.bit_length())
        mybin = lambda x: bin(x)[2:].zfill(blength)
        gcs = gray_code(a, b)
        # print(gcs)
        for m, n in zip(gcs, gcs[1:]):
            # print(mybin(m), mybin(n))
            assert mybin(m ^ n).count('1') == 1


def test_control_bits():
    for _ in range(10):
        core = [random.randint(10, 100) for _ in range(random.randint(2, 3))]
        # print(core)
        blength = max(i.bit_length() for i in core)
        gcb = control_bits(blength, core)
        bitmatrix = np.array([list(bin(i)[2:].zfill(blength)) for i in core])
        # print(bitmatrix)
        expected = [bool(int(bitmatrix[0, i])) if len(set(bitmatrix[:, i])) == 1 else None for i in range(blength)]
        assert gcb == tuple(expected), f'gcb {gcb} != expected {expected}'


def assert_gray(gc: Iterable):
    for m, n in zip(gc, gc[1:]):
        assert bin(m ^ n).count('1') == 1


def test_cogray_code():
    for _ in range(10):
        nums = list(range(100))
        ns = random.sample(nums, k=2)
        ms = random.sample(nums, k=2)
        # print(f'ns={ns}, ms={ms}')
        gc1, gc2 = cogray_code(ns, ms)
        for gc in gc1 + gc2:
            assert_gray(gc)

        blength = max(ns + ms).bit_length()
        mybin = lambda x: bin(x)[2:].zfill(blength)
        bits = np.array([list(mybin(c[-1])) for c in gc1 + gc2])
        # print(bits)
        bitdiff = [len(set(np.squeeze(bits[:, i]))) > 1 for i in range(bits.shape[1])]
        assert 1 == sum(bitdiff)
        i = bitdiff.index(True)
        assert len(set(np.squeeze(bits[:2, i]))) == 2, f'bits violates parity: {np.squeeze(bits[:2, i])}'
        assert len(set(np.squeeze(bits[2:, i]))) == 2, f'bits violates parity: {np.squeeze(bits[2:, i])}'
