import cirq

if __name__ == '__main__':
    q0 = cirq.LineQubit(0)
    qbs = [cirq.LineQubit.range(3) for _ in range(3)]
    circuit = cirq.Circuit(
        cirq.X(q0),
        cirq.H(q1),
        cirq.CCX(q0, q1, q2),
        cirq.M(q2),
        cirq.M(q1),
    )
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=100)
    print(circuit)
    print(result)
