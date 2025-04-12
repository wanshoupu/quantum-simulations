import cirq
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from common.utils.cnot_decompose import permeye, cnot_decompose
from common.utils.format_matrix import MatrixFormatter
from matplotlib import pyplot as plt
from qiskit.circuit.library import UnitaryGate


def cyclic_matrix(n, i):
    indexes = list(range(n))
    xs = indexes[:i] + np.roll(indexes[i:], 1).tolist()
    m = permeye(xs)
    return m


if __name__ == '__main__':
    formatter = MatrixFormatter()
    m = cyclic_matrix(8, 1)
    print(formatter.tostr(m))
    op = Operator(m)
    gate = UnitaryGate(m)
    qc = QuantumCircuit(3)
    qc.append(gate, [0, 1, 2])
    qc_decomposed = qc.decompose(reps=2)

    qc_decomposed.draw(output='mpl')
    plt.show()
