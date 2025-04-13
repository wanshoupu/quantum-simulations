from typing import Tuple

import numpy as np
from attr import dataclass
from numpy.typing import NDArray

from common.utils.format_matrix import MatrixFormatter
from common.utils.mgen import random_unitary


@dataclass
class Unitary2l:
    """
    Instantiate a 2-level unitary matrix.
    :param dimension: dimension of the matrix.
    :param matrix: the 2 x 2 core matrix.
    :param indexes: the indexes occupied by the 2 x 2 core submatrix.
    """
    dimension: int
    matrix: NDArray
    indexes: Tuple[int, int]

    def inflate(self) -> NDArray:
        i = self.controls.index(None)
        length = len(self.controls)
        if i == 0:
            result = self.matrix
            for _ in range(1, length):
                result = np.kron(result, np.eye(2))
            return result
        elif i == length:
            result = self.matrix
            for _ in range(1, length):
                result = np.kron(np.eye(2), result)
            return result
        # else
        result = np.eye(2)
        for _ in range(i):
            result = np.kron(result, np.eye(2))
        result = np.kron(result, self.matrix)
        for _ in range(i + 1, length):
            result = np.kron(result, np.eye(2))
        return result


class CUnitary(Unitary2l):
    def __init__(self, m: NDArray, controls: Tuple[bool]):
        """
        Instantiate a controlled single-qubit unitary matrix.
        :param m: the 2 x 2 core matrix.
        :param controls: the control qubit together with the 0(False) and 1 (True) state to actuate the control. There should be exactly one None state which is the target qubit.
        Dimension of the matrix is given by len(controls).
        """
        super().__init__(1 << len(controls), m, CUnitary.target_indexes(controls))
        self.controls = controls

    @staticmethod
    def target_indexes(controls: Tuple[bool]) -> Tuple[int, int]:
        i = controls.index(None)
        bits = list(controls)
        bits[i] = False
        num = lambda v: int(''.join('1' if (bits[i] != v) else '0' for v in bits), 2)
        return num(False), num(True)


if __name__ == '__main__':
    m = random_unitary(2)
    formatter = MatrixFormatter()
    cu = CUnitary(m, [True, True, None])

    print(cu)
    print(formatter.tostr(cu.inflate()))
