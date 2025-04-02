from qiskit import QuantumCircuit
from matplotlib import pyplot as plt
from qiskit.quantum_info import Operator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator

from common.simple_matrix import mformat

if __name__ == '__main__':
    # Create a new circuit with two qubits
    qc = QuantumCircuit(3)

    # Perform a controlled-X gate on qubit 1, controlled by qubit 0
    qc.cx(2, 0)
    unitary_matrix = Operator(qc).data
    print(mformat(unitary_matrix))

    # Return a drawing of the circuit using MatPlotLib ("mpl"). This is the
    # last line of the cell, so the drawing appears in the cell output.
    # Remove the "mpl" argument to get a text drawing.
    qc.draw(output='mpl')
    plt.show()
