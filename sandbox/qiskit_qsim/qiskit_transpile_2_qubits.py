from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import random_unitary
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.circuit.library import CXGate
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # This does not work and throws exception qiskit.transpiler.exceptions.TranspilerError: 'Unable to translate the operations in the circuit
    unitary = random_unitary(4)

    # Decompose unitary into CX + single-qubit gates (no opaque 'u' gates remain)
    decomposer = TwoQubitBasisDecomposer(CXGate())
    decomposed_circuit = decomposer(unitary)

    # Create a new QuantumCircuit and append the decomposed gates
    qc = QuantumCircuit(2)
    qc.append(decomposed_circuit, [0, 1])

    # Check the operations to confirm no 'u' gate remains:
    print(qc.count_ops())  # should show only known gates like 'cx', 'rz', 'ry', etc.

    # Transpile to your basis gates (e.g., Clifford+T)
    basis_gates = ['h', 's', 'sdg', 't', 'tdg', 'cx', 'x', 'y', 'z', 'id']
    transpiled_qc = transpile(qc, basis_gates=basis_gates, optimization_level=3)

    # Visualize
    transpiled_qc.draw('mpl')
    plt.savefig("qc_qiskit_sketch.pdf", bbox_inches='tight')
    plt.show()
