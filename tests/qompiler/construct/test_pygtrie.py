from pygtrie import Trie

from quompiler.construct.types import UnivGate


def test_trie_init():
    trie = Trie()
    trie[''] = [UnivGate.I]
    trie['H'] = [UnivGate.H]
    trie['XHX'] = [UnivGate.X, UnivGate.H, UnivGate.X]
    assert trie.keys() == [tuple(), ('H',), ('X', 'H', 'X')]


def test_trie_has_key():
    trie = Trie()
    trie[''] = [UnivGate.I]
    trie['H'] = [UnivGate.H]
    trie['XHX'] = [UnivGate.X, UnivGate.H, UnivGate.X]
    assert trie.has_key('XHX')


def test_trie_has_subtree():
    trie = Trie()
    trie[''] = [UnivGate.I]
    trie['H'] = [UnivGate.H]
    trie['XHX'] = [UnivGate.X, UnivGate.H, UnivGate.X]
    assert trie.has_subtrie('XH')


def test_trie_has_prefix():
    trie = Trie()
    trie[''] = [UnivGate.I]
    trie['H'] = [UnivGate.H]
    trie['XHX'] = [UnivGate.X, UnivGate.H, UnivGate.X]
    sp = trie.shortest_prefix('XHX')
    assert not sp.key
