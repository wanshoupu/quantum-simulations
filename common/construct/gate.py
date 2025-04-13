from abc import ABC
from dataclasses import dataclass
import numpy as np
from sympy.physics.quantum.gate import CNotGate
from numpy.typing import NDArray


class Gate(ABC):
    def __init__(self, m: NDArray):
        if CNotGate(m):
            pass
