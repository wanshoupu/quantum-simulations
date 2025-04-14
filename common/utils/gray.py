from typing import Iterable, Tuple, Optional

import numpy as np


def gray_code(n1, n2):
    """
    Generate Gray code between n1 and n2 (inclusive)
    :param n1: integer
    :param n2: integer
    :return: Gray code between n1 and n2 (inclusive). If n1 == n2, [n1] is returned.
    """
    result = [n1]
    for i in range(max(n1.bit_length(), n2.bit_length())):
        mask = 1 << i
        bit = n2 & mask
        if result[-1] & mask != bit:
            result.append((result[-1] ^ mask) | bit)
    return result


def control_bits(bitlength: int, core: Iterable) -> Tuple[Optional[bool], ...]:
    """
    Generate the control bits of a bundle of indexes given by core.
    The control bits are those bits shared by all the indexes in the core. The rest are target bits.
    The control bits are set to the corresponding common bits in the core (0->False, 1->True) whereas the target bit set to None.
    Big endian is used, namely, most significant bits on the left most end of the array.
    :param bitlength: total length of the control bits
    :param core: the core indexes, i.e., the indexes of the target bits
    :return: Tuple[bool] corresponding to the control bits
    """
    idiff = []
    for i in range(bitlength):
        mask = 1 << i
        if len({(a & mask) for a in core}) == 2:
            idiff.append(i)
    bits = [bool(core[0] & (1 << j)) for j in range(bitlength)]
    for i in idiff:
        bits[i] = None
    return tuple(bits[::-1])


if __name__ == '__main__':
    from common.utils.format_matrix import MatrixFormatter
    import random

    random.seed(5)
    formatter = MatrixFormatter()


    def _test_gray_code():
        for _ in range(10):
            a = random.randint(10, 100)
            b = random.randint(10, 100)
            # print(f'{a}, {b}')
            blength = max(a.bit_length(), b.bit_length())
            mybin = lambda x: bin(x)[2:].zfill(blength)
            gcs = gray(a, b)
            # print(gcs)
            for m, n in zip(gcs, gcs[1:]):
                # print(mybin(m), mybin(n))
                assert mybin(m ^ n).count('1') == 1


    def _test_control_bits():
        for _ in range(10):
            core = [random.randint(10, 100) for _ in range(random.randint(2, 3))]
            # print(core)
            blength = max(i.bit_length() for i in core)
            gcb = control_bits(blength, core)
            bitmatrix = np.array([list(bin(i)[2:].zfill(blength)) for i in core])
            # print(bitmatrix)
            expected = [bool(int(bitmatrix[0, i])) if len(set(bitmatrix[:, i])) == 1 else None for i in range(blength)]
            assert gcb == tuple(expected), f'gcb {gcb} != expected {expected}'
