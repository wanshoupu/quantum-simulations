import cirq
from cirq import merge_single_qubit_gates_to_phased_x_and_z, drop_negligible_operations, drop_empty_moments, eject_z

if __name__ == '__main__':
    # Create qubits
    q0 = cirq.NamedQubit("q0")
    q1 = cirq.NamedQubit("q1")

    # Build a sample circuit with redundant gates
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.H(q0),  # This cancels with the previous H
        cirq.CNOT(q0, q1),
        cirq.Z(q1),
        cirq.Z(q1),  # Two Z gates = identity (can be removed)
        cirq.measure(q0, q1)
    )

    print("Original circuit:")
    print(circuit)

    # Apply optimizers
    circuit = merge_single_qubit_gates_to_phased_x_and_z(circuit)
    circuit = eject_z(circuit)
    circuit = drop_negligible_operations(circuit)
    circuit = drop_empty_moments(circuit)

    print("\nOptimized circuit:")
    print(circuit)
