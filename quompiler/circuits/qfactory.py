from quompiler.circuits.cirq_factory.cirq_builder import CirqBuilder
from quompiler.circuits.qbuilder import CircuitBuilder
from quompiler.circuits.qdevice import QDevice
from quompiler.circuits.qiskit_factory.qiskit_builder import QiskitBuilder
from quompiler.circuits.qompiler import Qompiler
from quompiler.circuits.quimb_factory.quimb_builder import QuimbBuilder
from quompiler.circuits.render import QRenderer
from quompiler.config.construct import QompilerConfig
from quompiler.construct.types import QompilePlatform
from quompiler.optimize.basic_optimizer import SlidingWindowOptimizer
from quompiler.optimize.optimizer import Optimizer


class QFactory:

    def __init__(self, config: QompilerConfig):
        self._config = config

    def get_config(self):
        return self._config

    def get_device(self) -> QDevice:
        return QDevice(self._config.device)

    def get_optimizers(self) -> list[Optimizer]:
        return [SlidingWindowOptimizer(2)]

    def get_qompiler(self) -> Qompiler:
        return Qompiler(self._config, self.get_device())

    def get_render(self, target: QompilePlatform) -> QRenderer:
        return QRenderer(self.get_builder(target))

    def get_builder(self, target: QompilePlatform) -> CircuitBuilder:
        if target == QompilePlatform.CIRQ:
            return CirqBuilder()
        if target == QompilePlatform.QISKIT:
            return QiskitBuilder()
        if target == QompilePlatform.QUIMB:
            return QuimbBuilder()
        raise ValueError(f"Unsupported platform {target}")
