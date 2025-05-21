from abc import ABC, abstractmethod

from quompiler.construct.bytecode import Bytecode
from quompiler.construct.types import OptimizerLevel


class Optimizer(ABC):

    @abstractmethod
    def level(self) -> OptimizerLevel:
        pass

    @abstractmethod
    def optimize(self, code: Bytecode) -> Bytecode:
        pass
