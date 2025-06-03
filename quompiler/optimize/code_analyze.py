from collections import Counter, namedtuple
from typing import Sequence

from quompiler.construct.bytecode import Bytecode
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.qspace import Qubit

# Named tuple to hold stats
CodeMetadata = namedtuple(
    "Stats", [
        "num_nodes",
        "num_leaves",
        "height",
        "fanout_distr",
        "depth_distr",
        "num_nonstd",
        "qubit_distr",
    ])


def is_std(node: Bytecode) -> bool:
    if not node.is_leaf():
        return True
    data = node.data
    if not isinstance(data, CtrlGate):
        return False
    return data.is_std()


def ctrl_len(node: Bytecode) -> int:
    if not node.is_leaf():
        return -1
    data = node.data
    if not isinstance(data, CtrlGate):
        return -1
    return len(data.controls)


def extract_qspace(stats: CodeMetadata) -> Sequence[Qubit]:
    qspace = []
    for q in stats.qubit_distr:
        prefix = q[0]
        qid = int(q[1:])
        is_ancilla = True if prefix == 'a' else False
        qspace.append(Qubit(qid, ancilla=is_ancilla))
    return sorted(qspace)


def gen_stats(node: Bytecode) -> CodeMetadata:
    """
    collect tree statistics recursively
    """
    if node is None:
        return CodeMetadata(num_nodes=0,
                            num_leaves=0,
                            height=0,
                            fanout_distr=Counter(),
                            depth_distr=Counter(),
                            num_nonstd=0,
                            qubit_distr=Counter(),
                            )

    num_nodes = 1
    num_leaves = 1 if node.is_leaf() else 0
    num_nonstd = 0 if is_std(node) else 1
    fanout_distr = Counter({len(node.children): 1})
    depth_distr = Counter() if node.children else Counter({0: 1})
    qubit_distr = Counter({str(q): 1 for q in node.data.qspace}) if isinstance(node.data, CtrlGate) else Counter()

    height = 0
    for child in node.children:
        stats = gen_stats(child)
        num_nodes += stats.num_nodes
        num_leaves += stats.num_leaves
        num_nonstd += stats.num_nonstd
        fanout_distr.update(stats.fanout_distr)
        depth_distr.update(stats.depth_distr)
        if stats.height > height:
            height = stats.height
        qubit_distr.update(stats.qubit_distr)
    return CodeMetadata(
        num_nodes=num_nodes,
        num_leaves=num_leaves,
        height=(1 + height),
        fanout_distr=fanout_distr,
        depth_distr=Counter({k + 1: v for k, v in depth_distr.items()}),
        num_nonstd=num_nonstd,
        qubit_distr=qubit_distr,
    )


def percentile(freq: Counter, per: int):
    """
    Returns the value at the given percentile (0-100) from a Counter.
    Assumes the percentile is inclusive: e.g. percentile=50 gives median.
    """
    assert 0 <= per <= 100, "percentile must be between 0 and 100"
    assert freq
    total = sum(freq.values())
    assert total > 0, "Frequency total must be greater than 0."

    # Position in the sorted data (1-based rank)
    rank = (per / 100) * total
    sorted_items = sorted(freq.items())  # sort by keys

    cumulative = 0
    for key, freq in sorted_items:
        cumulative += freq
        if cumulative >= rank:
            return key

    return sorted_items[-1][0]  # fallback to last key


def average(freq: Counter):
    return sum(k * v for k, v in freq.items()) / sum(freq.values())


def print_stats(stats):
    print("Bytecode Statistics:")
    print(f"- Total Nodes: {stats.num_nodes}")
    print(f"- Leaf Nodes: {stats.num_leaves}")
    print(f"- Non-standard Gates: {stats.num_nonstd}")
    print(f"- Tree Height: {stats.height}")
    print()
    print(f"- Degree Distribution: {dict(stats.fanout_distr.most_common())}")
    print(f"- Max Degree: {max(stats.fanout_distr)}")
    print(f"- Average Degree: {average(stats.fanout_distr) :.2f}")
    print(f"- Median Degree: {percentile(stats.fanout_distr, 50) :.2f}")
    print()
    print(f"- Depth Distribution: {dict((stats.depth_distr.most_common()))}")
    print(f"- Max Depth: {max(stats.depth_distr)}")
    print(f"- Average Depth: {average(stats.depth_distr) :.2f}")
    print(f"- Median Depth: {percentile(stats.depth_distr, 50) :.2f}")
