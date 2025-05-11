from quompiler.construct.std_gate import CtrlStdGate
from quompiler.construct.types import EmitType, UnivGate
from quompiler.qompile.quompiler import granularity
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_UnitaryM, random_indexes, random_CtrlGate, random_control

formatter = MatrixFormatter(precision=2)


def test_granularity_UNITARY():
    n = 2
    dim = 1 << n
    indexes = random_indexes(dim, dim)
    u = random_UnitaryM(dim, indexes)
    grain = granularity(u)
    assert grain == EmitType.UNITARY


def test_granularity_TWO_LEVEL():
    n = 2
    dim = 1 << n
    indexes = random_indexes(dim, 2)
    u = random_UnitaryM(dim, indexes)
    grain = granularity(u)
    assert grain == EmitType.TWO_LEVEL


def test_granularity_MULTI_TARGET():
    n = 2
    ctrl = random_control(n, n)
    u = random_CtrlGate(ctrl)
    grain = granularity(u)
    assert grain == EmitType.MULTI_TARGET


def test_granularity_SINGLET():
    n = 4
    ctrl = random_control(n, 1)
    u = random_CtrlGate(ctrl)
    grain = granularity(u)
    assert grain == EmitType.SINGLET


def test_granularity_TWO_CTRL():
    n = 3
    ctrl = random_control(n, 1)
    u = random_CtrlGate(ctrl)
    grain = granularity(u)
    assert grain == EmitType.TWO_CTRL


def test_granularity_ONE_CTRL():
    n = 2
    ctrl = random_control(n, 1)
    u = random_CtrlGate(ctrl)
    grain = granularity(u)
    assert grain == EmitType.ONE_CTRL


def test_granularity_std_TWO_CTRL():
    """
    This test demo that CtrlStdGate are never classified as EmitType.TWO_CTRL or EmitType.ONE_CTRL
    """
    n = 3
    ctrl = random_control(n, 1)
    u = CtrlStdGate(UnivGate.Y, ctrl)
    grain = granularity(u)
    assert grain == EmitType.UNIV_GATE


def test_granularity_std_ONE_CTRL():
    """
    This test demo that CtrlStdGate are never classified as EmitType.TWO_CTRL or EmitType.ONE_CTRL
    """
    n = 2
    ctrl = random_control(n, 1)
    u = CtrlStdGate(UnivGate.H, ctrl)
    grain = granularity(u)
    assert grain == EmitType.CLIFFORD_T
