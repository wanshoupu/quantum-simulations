import pytest

from quompiler.construct.types import EmitType, UnivGate
from quompiler.optimize.window import ConsolidateOperator
from quompiler.utils.mgen import create_bytecode


@pytest.mark.parametrize("seq, ctrl_num, emit, count, expected", [
    ['T,SD', 3, EmitType.SINGLET, 1, 'TD'],
    ['T,SD,S,T', 2, EmitType.CLIFFORD_T, 1, 'S'],
    ["S,T,SD", 1, EmitType.CLIFFORD_T, 1, 'T'],
    ["S,T,SD,TD", 1, EmitType.CLIFFORD_T, 1, 'I'],
])
def test_consolidate_operator(seq, ctrl_num, emit, count, expected):
    nodes = create_bytecode(seq, ctrl_num).children
    length = len(nodes)
    opt = ConsolidateOperator(length, emit=emit)
    for code in nodes:
        opt.run(code)
    remains = [n for n in nodes if not n.skip]
    assert len(remains) == count
    actual = remains[0].data.gate
    assert actual == UnivGate[expected]
