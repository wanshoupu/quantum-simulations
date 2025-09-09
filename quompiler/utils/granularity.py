from typing import Union

from quompiler.construct.cgate import CtrlGate
from quompiler.construct.types import GateGrain, UnivGate
from quompiler.construct.unitary import UnitaryM


def granularity(obj: Union[UnitaryM, CtrlGate]) -> GateGrain:
    """
    Given an input obj, determine the granularity level if we are to return it as is.
    The granularity is provided in terms of EmitType (see quompiler.construct.types.EmitType).
    :param obj: input object of one of the types
    :return: EmitType denoting the granularity
    """
    if isinstance(obj, UnitaryM):
        return granularity_mat(obj)
    if isinstance(obj, CtrlGate):
        return granularity_gate(obj)
    return GateGrain.INVALID


def granularity_mat(obj: UnitaryM) -> GateGrain:
    if obj.is2l():
        return GateGrain.TWO_LEVEL
    return GateGrain.UNITARY


def granularity_gate(obj: CtrlGate) -> GateGrain:
    if obj.is_std():
        if 1 < len(obj.control_qids()):
            return GateGrain.SINGLET
        if obj.gate in UnivGate.cliffordt():
            return GateGrain.CLIFFORD_T
        return GateGrain.CTRL_PRUNED

    if obj.is_principal():
        return GateGrain.PRINCIPAL

    # not std, not principal
    if not obj.issinglet():
        return GateGrain.MULTI_TARGET
    if len(obj.control_qids()) < 2:
        return GateGrain.CTRL_PRUNED
    return GateGrain.SINGLET
