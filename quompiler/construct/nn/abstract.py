from abc import ABC, abstractmethod

from numpy.typing import NDArray

from quompiler.construct.bytecode import Bytecode
from quompiler.construct.types import UnivGate, SU2NetType
from quompiler.utils.std_decompose import cliffordt_seqs


class QNN(ABC):
    @abstractmethod
    def __init__(self, seq: list[tuple[NDArray, tuple[UnivGate]]]):
        pass

    @abstractmethod
    def lookup(self, mat: NDArray) -> tuple[Bytecode, float]:
        pass


def create_qnn(length: int, impl_type: SU2NetType) -> QNN:
    if impl_type == SU2NetType.KDTreeNN:
        seqs = cliffordt_seqs(length)
        from quompiler.construct.nn.kdtree import KDTreeNN
        return KDTreeNN(seqs)
    if impl_type == SU2NetType.BruteNN:
        seqs = cliffordt_seqs(length)
        from quompiler.construct.nn.nn import BruteNN
        return BruteNN(seqs)
    if impl_type == SU2NetType.AutoNN:
        seqs = cliffordt_seqs(length)
        from quompiler.construct.nn.nn import AutoNN
        return AutoNN(seqs)
    if impl_type == SU2NetType.BallTreeNN:
        seqs = cliffordt_seqs(length)
        from quompiler.construct.nn.nn import BallTreeNN
        return BallTreeNN(seqs)

    raise NotImplementedError(f'SU2NetType.{impl_type} is not implemented')
