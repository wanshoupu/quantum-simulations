from abc import ABC, abstractmethod

from quompiler.construct.bytecode import Bytecode
from quompiler.construct.types import OptLevel


class Optimizer(ABC):

    @abstractmethod
    def level(self) -> OptLevel:
        pass

    @abstractmethod
    def optimize(self, code: Bytecode) -> Bytecode:
        pass
