from pygtrie import Trie

from quompiler.construct.types import UnivGate


class SeqOptimizer:
    def __init__(self):
        self.trie = Trie()

    def optimized(self, mat, seq) -> list:
        key = ''.join(g.name for g in seq)
        for i in range(2, 1 + len(key)):
            suffix = key[-i:]
            if self.trie.has_key(suffix):
                return seq[:i] + self.trie[suffix]
            gate = UnivGate.get_prop(mat)
            if gate is not None:
                self.trie[suffix] = [gate]
                return seq[:i] + [gate]
        return seq
