import cirq
import numpy as np

from common.utils.cnot_decompose import permeye, cnot_decompose
from common.utils.format_matrix import MatrixFormatter
from common.utils.mat2l_decompose import mat2l_decompose


def cyclic_matrix(n, i):
    indexes = list(range(n))
    xs = indexes[:i] + np.roll(indexes[i:], 1).tolist()
    m = permeye(xs)
    return m


if __name__ == '__main__':
    formatter = MatrixFormatter()
    m = cyclic_matrix(8, 1)
    print(f'cyclic matrix:\n{formatter.tostr(m)}\n')
    m2ls = mat2l_decompose(m)
    result = []
    for x in m2ls:
        coms = cnot_decompose(x)
        cnots, u = coms[:-1], coms[-1]
        print(f'single-qubit unitary matrix:\n{formatter.tostr(u)}\n')
        result.extend(coms)

    for x in result:
        print(f'gate-level matrix:\n{formatter.tostr(x)}\n')

    # Define 3 qubits
    q0, q1, q2 = cirq.LineQubit.range(3)

    # Create a circuit with an operation only on q0 and q1
    circuit = cirq.Circuit([
        cirq.CX(q2, q0),
        cirq.CX(q2, q0),
    ])

    # print(formatter.mformat(circuit.unitary(qubit_order=[q0, q1, q2])))
    #
    # # Explicitly specify all qubits when printing
    # print(circuit.to_text_diagram(qubit_order=[q0, q1, q2]))
