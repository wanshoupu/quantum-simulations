from common.construct.quompiler import quompile
from common.utils.mgen import cyclic_matrix


def test_compile():
    u = cyclic_matrix(8, 1)
    bc = quompile(u)
    print(bc)
