import json
import random

from quompiler.construct.bytecode import Bytecode
from quompiler.optimize.code_analyze import gen_stats
from quompiler.utils.mgen import random_UnitaryM, random_indexes, random_ctrlgate


def random_code() -> Bytecode:
    n = random.randint(1, 5)
    dim = 1 << n
    indexes = random_indexes(dim, dim)
    children = [Bytecode(random_ctrlgate(random.randint(3, 5), random.randint(1, 3), 10000)) for _ in range(3)]
    root = Bytecode(random_UnitaryM(dim, indexes), children)
    return root


def test_gen_stats():
    code = random_code()
    # Print the statistics
    stats = gen_stats(code)
    sstats = json.dumps(stats)
    print(sstats)
    assert len(sstats) == 150
