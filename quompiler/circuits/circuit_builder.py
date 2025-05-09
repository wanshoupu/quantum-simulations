from abc import ABC, abstractmethod
from typing import Optional, Union

from quompiler.construct.cgate import CtrlGate
from quompiler.construct.unitary import UnitaryM
from quompiler.construct.types import UnivGate


class CircuitBuilder(ABC):

    @abstractmethod
    def __init__(self, dimension: int):
        """
        Create a circuit builder of a linear array of qubits.
        :param dimension: the total number of qubits.
        """
        pass

    @abstractmethod
    def build_gate(self, m: Union[UnitaryM, CtrlGate]) -> None:
        """
        Build a unitary gate out of the matrix m
        :param m: UnitaryM possibly with control bits
        """
        pass

    @abstractmethod
    def finish(self, optimized=False) -> object:
        """
        Call this method after building process is done to retrieve what's built.
        It's expected that the building process is incremental meaning that
        this method may be called multiple times while building process is en route.
        :return:
        """
        pass

    @abstractmethod
    def get_univ_gate(self, m: Union[UnitaryM, CtrlGate]) -> Optional[UnivGate]:
        """
        Subclass return a universal gate out of a set, which is to be used as the building blocks.
        :return: The universal gate, if any, for the input m. Return None if not found.
        """
        pass

    def build_group(self, m: Union[UnitaryM, CtrlGate]):
        """
        This method allows builder to group multiple gates together to represent, e.g., a hierarchy.
        Overriding this is optional for subclasses.
        :param m: the root UnitaryM for this group.
        """
        pass
