import cirq
from cirq.contrib.svg import circuit_to_svg

if __name__ == '__main__':
    # Define qubits
    q0, q1 = cirq.LineQubit.range(2)

    # Create a quantum circuit
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, q1)
    )

    # Generate SVG representation of the circuit
    svg = circuit_to_svg(circuit)

    # Save the SVG to a file
    with open("circuit.svg", "w") as f:
        f.write(svg)
