from abc import ABC, abstractmethod
from typing import Union, Optional

from quompiler.config.construct import DeviceConfig
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.types import UnivGate
from quompiler.construct.unitary import UnitaryM


class CircuitBuilder(ABC):

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
    def all_qubits(self) -> list:
        """
        Return a list of all possible qubits in orginal sorting order.
        """
        pass

    def build_group(self, m: Union[UnitaryM, CtrlGate]):
        """
        This method allows builder to group multiple gates together to represent, e.g., a hierarchy.
        Overriding this is optional for subclasses.
        :param m: the root UnitaryM for this group.
        """
        pass

    @abstractmethod
    def register(self, qspace):
        pass
