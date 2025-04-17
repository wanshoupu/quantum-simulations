import cirq
from cirq import CNotPowGate, CXPowGate

from quompiler.utils.format_matrix import MatrixFormatter


def cirq_mat():
    gate = cirq.CXPowGate(exponent=1)

    # Get the unitary matrix
    matrix = cirq.unitary(gate)
    print(matrix)


if __name__ == '__main__':
    # 2-qubit unitary matrix: shape (4, 4)
    import numpy as np
    U2 = np.eye(4)  # identity for example

    gate2 = cirq.MatrixGate(U2)
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(gate2(q0, q1))

    print(circuit)
