import cirq

# Create a simple circuit
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, key='c0'),
    cirq.measure(q1, key='c1')
)

# Export to QASM 2.0
qasm_output = cirq.qasm(circuit)
if __name__ == '__main__':

    print(qasm_output)
