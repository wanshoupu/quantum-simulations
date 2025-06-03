from abc import abstractmethod, ABC

from quompiler.circuits.qbuilder import CircuitBuilder
from quompiler.circuits.qdevice import QDevice
from quompiler.circuits.qompiler import Qompiler
from quompiler.circuits.render import QRenderer
from quompiler.config.construct import QompilerConfig
from quompiler.optimize.optimizer import Optimizer


class QFactory(ABC):

    def __init__(self, config: QompilerConfig):
        self._config = config
        self._qompiler = None

    def get_config(self):
        return self._config

    def get_qompiler(self) -> Qompiler:
        if self._qompiler is None:
            self._qompiler = Qompiler(self._config, self.get_builder(), self.get_device(), self.get_optimizers())
        return self._qompiler

    @abstractmethod
    def get_device(self) -> QDevice:
        pass

    @abstractmethod
    def get_builder(self) -> CircuitBuilder:
        pass

    @abstractmethod
    def get_optimizers(self) -> list[Optimizer]:
        pass

    @abstractmethod
    def get_render(self) -> QRenderer:
        pass
