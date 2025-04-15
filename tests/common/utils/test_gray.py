from common.utils.format_matrix import MatrixFormatter
import random
import numpy as np

from common.utils.gray import gray_code, control_bits

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
