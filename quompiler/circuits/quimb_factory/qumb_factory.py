from typing_extensions import  override

from quompiler.circuits.abstract_factory import QFactory
from quompiler.circuits.qbuilder import CircuitBuilder
from quompiler.circuits.qdevice import QDevice
from quompiler.circuits.qompiler import Qompiler
from quompiler.config.construct import QompilerConfig


class QuimbFactory(QFactory):
    def __init__(self, config: QompilerConfig):
        self._config = config

    @override
    def get_qompiler(self) -> Qompiler:
        pass

    @override
    def get_device(self) -> QDevice:
        pass

    @override
    def get_builder(self) -> CircuitBuilder:
        pass
