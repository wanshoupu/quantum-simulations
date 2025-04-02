import cirq

from common.simple_matrix import mformat

if __name__ == '__main__':
    # Define 3 qubits
    q0, q1, q2 = cirq.LineQubit.range(3)

    # Create a circuit with an operation only on q0 and q1
    circuit = cirq.Circuit(
        cirq.CX(q2, q0)
    )

    print(mformat(circuit.unitary(qubit_order=[q0, q1, q2])))

    # Explicitly specify all qubits when printing
    print(circuit.to_text_diagram(qubit_order=[q0, q1, q2]))
