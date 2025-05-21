from typing_extensions import override

from quompiler.construct.bytecode import Bytecode
from quompiler.construct.types import OptimizerLevel
from quompiler.optimize.optimizer import Optimizer


class SlidingWindowOptimizer(Optimizer):
    """
    Sliding window optimizer performs optimization on adjacent moments within a sliding window of certain size.
    Example optimization performed:
    - Replace subsequence HXH with Z, HYH with -1 and Y, etc.
    - Replace two consecutive T by S
    - Replace two consecutive S by Z
    - Replace two consecutive X by I
    - Eliminate I
    """

    def __init__(self, window: int):
        self.window = window

    @override
    def level(self) -> OptimizerLevel:
        return OptimizerLevel.O0

    @override
    def optimize(self, code: Bytecode) -> Bytecode:
        return code
