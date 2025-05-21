from typing import Union

from quompiler.construct.cgate import CtrlGate
from quompiler.construct.types import EmitType, UnivGate
from quompiler.construct.unitary import UnitaryM


def granularity(obj: Union[UnitaryM, CtrlGate]) -> EmitType:
    """
    Given an input obj, determine the granularity level if we are to return it as is.
    The granularity is provided in terms of EmitType (see quompiler.construct.types.EmitType).
    :param obj: input object of one of the types
    :return: EmitType denoting the granularity
    """
    if isinstance(obj, UnitaryM):
        if obj.is2l():
            return EmitType.TWO_LEVEL
        return EmitType.UNITARY

    if isinstance(obj, CtrlGate) and not obj.is_std():
        if not obj.issinglet():
            return EmitType.MULTI_TARGET
        if len(obj.control_qids()) < 2:
            return EmitType.CTRL_PRUNED
        return EmitType.SINGLET

    if isinstance(obj, CtrlGate) and obj.is_std():
        if 1 < len(obj.control_qids()):
            return EmitType.SINGLET
        if obj.gate in UnivGate.cliffordt():
            return EmitType.CLIFFORD_T
        return EmitType.CTRL_PRUNED

    return EmitType.INVALID
