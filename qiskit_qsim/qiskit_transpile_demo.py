from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, transpile

from common.utils.mgen import random_unitary

if __name__ == '__main__':
    # Create a circuit with a complex gate
    qc = QuantumCircuit(3)
    u = random_unitary(8)
    qc.unitary(u, list(range(3)))

    # Decompose the circuit into U and CX gates
    decomposed_qc = qc.decompose()

    # Decompose to use only u3 and cx gates
    basis_gates = ['h', 's', 'sdg', 't', 'tdg', 'cx', 'x', 'y', 'z', 'id', 'u3']

    transpiled_qc = transpile(qc, basis_gates=basis_gates)

    print(transpiled_qc.draw('mpl'))

    # # Print the decomposed circuit
    # print(decomposed_qc)
    # decomposed_qc.draw('mpl')
    plt.show()
