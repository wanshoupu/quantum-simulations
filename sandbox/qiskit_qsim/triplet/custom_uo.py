import qiskit
import numpy as np
from qiskit.circuit.library import UnitaryGate
from matplotlib import pyplot as plt

from quompiler.utils.mgen import random_unitary

if __name__ == '__main__':
    # Define a 3-qubit unitary matrix (example)
    unitary_matrix = random_unitary(8)

    # Create a quantum circuit with 3 qubits
    qc = qiskit.QuantumCircuit(3)

    # Create a unitary gate from the matrix
    unitary_gate = UnitaryGate(unitary_matrix, label="CNOT31")

    # Apply the gate to the qubits
    qc.append(unitary_gate, [0, 1, 2])

    # Draw the circuit (optional)
    qc.draw(output='mpl')
    plt.show()
