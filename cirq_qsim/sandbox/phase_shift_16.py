import cirq

from common.utils.format_matrix import MatrixFormatter

if __name__ == '__main__':
    q0 = cirq.NamedQubit('q')

    # Create a circuit with an operation only on q0 and q1
    circuit = cirq.Circuit(
        [cirq.H(q0), cirq.T(q0), cirq.H(q0), cirq.T(q0) ** -1,
         cirq.H(q0), cirq.T(q0), cirq.H(q0), cirq.T(q0) ** -1,
         cirq.H(q0), cirq.T(q0), cirq.H(q0)]
    )

    formatter = MatrixFormatter(precision=3)
    print(formatter.mformat(circuit.unitary(qubit_order=[q0])))

    # Explicitly specify all qubits when printing
    print(circuit.to_text_diagram(qubit_order=[q0]))
