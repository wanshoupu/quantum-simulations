from matplotlib import pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import sys
import platform
import qiskit
from quompiler.utils.format_matrix import MatrixFormatter

if __name__ == '__main__':
    # Create a new circuit with two qubits
    qc = QuantumCircuit(3)
    print(f"Python version: {sys.version.split()[0]}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Qiskit version: {qiskit.__version__}")

    # Perform a controlled-X gate on qubit 0, controlled by qubit 2
    formatter = MatrixFormatter()
    qc.cx(2, 0)
    unitary_matrix = Operator(qc).data
    print(formatter.mformat(unitary_matrix))
    # There is an issue with the unitary matrix. Expected:
    # [[1 0 0 0 0 0 0 0]
    #  [0 0 0 0 0 1 0 0]
    #  [0 0 1 0 0 0 0 0]
    #  [0 0 0 0 0 0 0 1]
    #  [0 0 0 0 1 0 0 0]
    #  [0 1 0 0 0 0 0 0]
    #  [0 0 0 0 0 0 1 0]
    #  [0 0 0 1 0 0 0 0]]
    # got
    # [[1 0 0 0 0 0 0 0]
    #  [0 1 0 0 0 0 0 0]
    #  [0 0 1 0 0 0 0 0]
    #  [0 0 0 1 0 0 0 0]
    #  [0 0 0 0 0 1 0 0]
    #  [0 0 0 0 1 0 0 0]
    #  [0 0 0 0 0 0 0 1]
    #  [0 0 0 0 0 0 1 0]]

    # Return a drawing of the circuit using MatPlotLib ("mpl"). This is the
    # last line of the cell, so the drawing appears in the cell output.
    # Remove the "mpl" argument to get a text drawing.
    qc.draw(output='mpl', reverse_bits=True)
    plt.show()
