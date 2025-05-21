from quompiler.circuits.factory_manager import FactoryManager
from quompiler.utils.mgen import random_unitary
from collections import Counter, namedtuple

from quompiler.construct.bytecode import Bytecode


def analyze(root: Bytecode) -> bool:
    if root is None:
        return False
    print()
    print(root.metadata)
    if root.is_leaf() and len(root.data.controls) > 1:
        return True
    return any(analyze(c) for c in root.children)


# Named tuple to hold stats
Stats = namedtuple(
    "Stats", ["num_nodes", "num_leaves",
              "height", "trunk", "degree_freq", "depth_freq"])


# collect tree statistics recursively
def tree_stats(node):
    if node is None:
        return Stats(num_nodes=0, num_leaves=0, height=0,
                     trunk=[], degree_freq=Counter(), depth_freq=Counter())

    num_nodes = 1
    num_leaves = 0 if node.children else 1
    degree_freq = Counter({len(node.children): 1})
    depth_freq = Counter() if node.children else Counter({0: 1})

    height, trunk = 0, []
    for child in node.children:
        stats = tree_stats(child)
        num_nodes += stats.num_nodes
        num_leaves += stats.num_leaves
        degree_freq.update(stats.degree_freq)
        depth_freq.update(stats.depth_freq)
        if stats.height > height:
            height = stats.height
            trunk = stats.trunk
    return Stats(
        num_nodes=num_nodes,
        num_leaves=num_leaves,
        height=(1 + height),
        degree_freq=degree_freq,
        trunk=([node] + trunk),
        depth_freq=Counter({k + 1: v for k, v in depth_freq.items()})
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


def print_tree_stats(root):
    stats = tree_stats(root)
    print("Bytecode Statistics:")
    print(f"- Total Nodes: {stats.num_nodes}")
    print(f"- Leaf Nodes: {stats.num_leaves}")
    print(f"- Tree Height: {stats.height}")
    print(f"- Tree Trunk:")
    for i, node in enumerate(stats.trunk):
        print(f"    {'root' if i == 0 else str(i)} - {node.metadata['data']}")
    print()
    print(f"- Degree Distribution: {dict(stats.degree_freq.most_common())}")
    print(f"- Max Degree: {max(stats.degree_freq)}")
    print(f"- Average Degree: {average(stats.degree_freq) :.2f}")
    print(f"- Median Degree: {percentile(stats.degree_freq, 50) :.2f}")
    print(f"- Depth Distribution: {dict((stats.depth_freq.most_common()))}")
    print(f"- Max Depth: {max(stats.depth_freq)}")
    print(f"- Average Depth: {average(stats.depth_freq) :.2f}")
    print(f"- Median Depth: {percentile(stats.depth_freq, 50) :.2f}")


if __name__ == '__main__':
    n = 5
    dim = 1 << n
    u = random_unitary(dim)

    factory = FactoryManager().create_factory()
    compiler = factory.get_qompiler()
    code = compiler.compile(u)
    # Print the statistics
    print_tree_stats(code)
