from quompiler.construct.std_gate import CtrlStdGate
from quompiler.construct.types import EmitType, UnivGate
from quompiler.qompile.quompiler import granularity
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_UnitaryM, random_indexes, random_CtrlGate, random_control, random_unitary

formatter = MatrixFormatter(precision=2)


def test_granularity_unitary():
    n = 2
    dim = 1 << n
    indexes = random_indexes(dim, dim)
    u = random_UnitaryM(dim, indexes)
    grain = granularity(u)
    assert grain == EmitType.UNITARY


def test_granularity_two_level():
    n = 2
    dim = 1 << n
    indexes = random_indexes(dim, 2)
    u = random_UnitaryM(dim, indexes)
    grain = granularity(u)
    assert grain == EmitType.TWO_LEVEL


def test_granularity_multi_target():
    n = 2
    ctrl = random_control(n, n)
    u = random_CtrlGate(ctrl)
    grain = granularity(u)
    assert grain == EmitType.MULTI_TARGET


def test_granularity_singlet():
    n = 2
    ctrl = random_control(n, 1)
    u = random_CtrlGate(ctrl)
    grain = granularity(u)
    assert grain == EmitType.SINGLET


def test_granularity_multi_ctrl():
    n = 3
    ctrl = random_control(n, 1)
    u = random_CtrlGate(ctrl)
    grain = granularity(u)
    assert grain == EmitType.CTRL_PRUNED


def test_granularity_std_ctrl_pruned():
    """
    This test demo that CtrlStdGate are classified as EmitType.CTRL_PRUNED
    """
    n = 3
    ctrl = random_control(n, 1)
    u = random_CtrlGate(ctrl)
    grain = granularity(u)
    assert grain == EmitType.CTRL_PRUNED


def test_granularity_std_univ_gate():
    n = 2
    ctrl = random_control(n, 1)
    u = CtrlStdGate(UnivGate.Y, ctrl)
    grain = granularity(u)
    assert grain == EmitType.UNIV_GATE


def test_granularity_std_clifford_t():
    n = 1
    ctrl = random_control(n, 1)
    u = CtrlStdGate(UnivGate.H, ctrl)
    grain = granularity(u)
    assert grain == EmitType.CLIFFORD_T


def test_granularity_invalid():
    n = 1
    dim = 1 << n
    mat = random_unitary(dim)
    grain = granularity(mat)
    assert grain == EmitType.INVALID
