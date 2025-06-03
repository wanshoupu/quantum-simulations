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

    def get_config(self):
        return self._config

    def get_qompiler(self) -> Qompiler:
        return Qompiler(self._config, self.get_builder(), self.get_device(), self.get_optimizers())

    def get_render(self) -> QRenderer:
        return QRenderer(self._config, self.get_builder())

    def get_device(self) -> QDevice:
        return QDevice(self._config.device)

    @abstractmethod
    def get_builder(self) -> CircuitBuilder:
        pass

    @abstractmethod
    def get_optimizers(self) -> list[Optimizer]:
        pass
