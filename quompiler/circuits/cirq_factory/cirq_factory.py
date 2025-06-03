from typing_extensions import override

from quompiler.circuits.abstract_factory import QFactory
from quompiler.circuits.cirq_factory.cirq_builder import CirqBuilder
from quompiler.circuits.qbuilder import CircuitBuilder
from quompiler.config.construct import QompilerConfig
from quompiler.optimize.basic_optimizer import SlidingWindowOptimizer
from quompiler.optimize.optimizer import Optimizer


class CirqFactory(QFactory):

    def __init__(self, config: QompilerConfig):
        super().__init__(config)

    @override
    def get_optimizers(self) -> list[Optimizer]:
        return [SlidingWindowOptimizer(2)]

    @override
    def get_builder(self) -> CircuitBuilder:
        return CirqBuilder()
