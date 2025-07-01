import qiskit.qasm3 as qasm
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit

from quompiler.utils.mgen import random_unitary

if __name__ == '__main__':
    # Create a circuit with a complex gate
    qc = QuantumCircuit(3)
    u = random_unitary(8)
    qc.unitary(u, list(range(3)))

    # Decompose the circuit into U and CX gates
    decomposed_qc = qc.decompose()


    # Export to QASM string
    qasm_code = qasm.dumps(decomposed_qc)

    # Print the QASM code
    print(qasm_code)
    decomposed_qc.draw('mpl')
    plt.show()
