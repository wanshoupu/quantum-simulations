import cirq

if __name__ == '__main__':
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.X(q0),
        cirq.H(q1),
        cirq.CCX(q0, q1, q2),
        cirq.M(q2),
        cirq.M(q1),
    )
    print(circuit)

    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=100)
    print(result)
