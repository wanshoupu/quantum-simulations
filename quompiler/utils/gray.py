from typing import Tuple, List


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


def cogray_code(ns: Tuple[int, int], ms: Tuple[int, int]) -> tuple[tuple, tuple]:
    """
    Find the co-Gray code for ns and ms such that the total length of the gray code chain is minimal.
    Co-Gray code of a group of N numbers (i1,i2,...) consists of N tuples each begins with the corresponding number in the group and ends on a common number.
    For our purpose, we cull the last number such that destination consists of two numbers that differ only on one bit.
    For example, suppose ns = (5,4), ms = (6,4). In binary forms, one co-Gray code is
    n1 101
    n2 100
    m1 110 100
    m2 100 101
    Therefore, the co-Gray code is
    ((5,), (4,), (6,4), (4,5))
    Another co-Gray path is
    n1 101
    n2 100
    m1 110 111 101
    m2 100
    So an alternative co-Gray code could be
    ((5,), (4,), (6,7,5),(4,))
    Since the total length of Gray chain in both cases are the same, they are equally acceptable.
    :param ns: a pair of integers, (n1,n2), ni >= 0, i = 1,2
    :param ms: a pair of integers, (m1,m2), mi >= 0, i = 1,2
    :return: the co-Gray code
    """
    assert ns[0] != ns[1], f'Clashing numbers are forbidden: ns={ns}.'
    assert ms[0] != ms[1], f'Clashing numbers are forbidden: ms={ms}.'
    gs = [ns[0]], [ns[1]], [ms[0]], [ms[1]]
    diffs, parity = diff_parity_bits(ns, ms)
    if parity:
        p = parity[0]
        diffs.remove(p)
    else:
        p = diffs.pop()
        mask = 1 << p
        if gs[0][-1] & mask == gs[1][-1] & mask:
            gs[1].append(gs[1][-1] ^ mask)
        if gs[2][-1] & mask == gs[3][-1] & mask:
            gs[3].append(gs[3][-1] ^ mask)

    # find the target bit where g1[-1] and g2[-1] differ and g3[-1] and g4[-1] differ
    for i in diffs:
        total = sum((gs[j][-1] >> i) & 1 for j in range(len(gs)))
        bit = 0 if total < 2 else 1
        mask = 1 << i
        for j in range(len(gs)):
            if (gs[j][-1] >> i) & 1 != bit:
                gs[j].append(gs[j][-1] ^ mask)
    return gs[:2], gs[2:]


def diff_parity_bits(ns: Tuple[int, int], ms: Tuple[int, int]) -> Tuple[List[int], List[int]]:
    """
    Parity bit indexes are those where the numbers in each group have differing bit at this index.
    For example, ns = (5,6), ms = (7,0), the parity bits are [0,1] because if written in binary
    5: 101
    6: 110
    7: 111
    0: 000
    Note we adopt the Little endian convention the least significant bit has index zero.
    :param ns:
    :param ms:
    :return: The indexes of the parity bits. If no such bit exists, return empty list
    """
    bitlength = max(ns + ms).bit_length()
    parity = []
    diffs = []
    for i in range(bitlength):
        mask = 1 << i
        if len({ns[0] & mask, ns[1] & mask}) > 1 and len({ms[0] & mask, ms[1] & mask}) > 1:
            parity.append(i)
        if len({ns[0] & mask, ns[1] & mask, ms[0] & mask, ms[1] & mask}) > 1:
            diffs.append(i)
    return diffs, parity


def differ_bits(ns: Tuple[int, int], ms: Tuple[int, int]) -> List[int]:
    """
    Find any index of bit where the numbers in each of ns and ms differ on this bit.
    For example, ns = (5,6), ms = (7,0), the parity bit is 0.
    Note we adopt the Little endian convention the least significant bit has index zero.
    :param ns:
    :param ms:
    :return: index of the parity bit. If no such bit exists, return None
    """
    bitlength = max(ns + ms).bit_length()
    result = []
    for i in range(bitlength):
        mask = 1 << i
        if len({ns[0] & mask, ns[1] & mask, ms[0] & mask, ms[1] & mask}) > 1:
            result.append(i)
    return result


