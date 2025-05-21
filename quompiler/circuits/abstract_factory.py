from abc import abstractmethod, ABC

from quompiler.circuits.qbuilder import CircuitBuilder
from quompiler.circuits.qdevice import QDevice
from quompiler.circuits.qompiler import Qompiler
from quompiler.optimize.optimizer import Optimizer


class QFactory(ABC):
    @abstractmethod
    def get_qompiler(self) -> Qompiler:
        pass

    @abstractmethod
    def get_device(self) -> QDevice:
        pass

    @abstractmethod
    def get_builder(self) -> CircuitBuilder:
        pass

    @abstractmethod
    def get_optimizers(self) -> list[Optimizer]:
        pass
