import cirq
import random

if __name__ == '__main__':
    n = 7
    qbs = cirq.LineQubit.range(n)
    ops = []
    for i in range(n):
        ops.append(cirq.X(qbs[i]) if random.randint(0, 1) else cirq.I(qbs[i]))
    for i in range(n):
        ops.append(cirq.H(qbs[i]))
        for j in range(i + 1, n):
            cp = cirq.ControlledGate(cirq.ZPowGate(exponent=1 / (1 << j)))
            ops.append(cp(qbs[j], qbs[i]))
    for i in range(n // 2):
        ops.append(cirq.SWAP(qbs[i], qbs[-(i + 1)]))
    for i in range(n):
        ops.append(cirq.M(qbs[i]))
    circuit = cirq.Circuit(*ops)
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=10)
    print(circuit)
    print(result)
