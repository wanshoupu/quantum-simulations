from qiskit import QuantumCircuit
from matplotlib import pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator

if __name__ == '__main__':
    # Create a new circuit with two qubits
    qc = QuantumCircuit(3)

    # Perform a controlled-X gate on qubit 1, controlled by qubit 0
    qc.cx(0, 1)

    # Return a drawing of the circuit using MatPlotLib ("mpl"). This is the
    # last line of the cell, so the drawing appears in the cell output.
    # Remove the "mpl" argument to get a text drawing.
    qc.draw(output='mpl')
    plt.show()
