import matplotlib.pyplot as plt

import cirq


def simple_circuit():
    global q0, q1, result
    q0, q1 = cirq.LineQubit.range(2)  # Create 2 qubits in a line
    circuit = cirq.Circuit(
        cirq.H(q0),  # Hadamard gate on q0
        cirq.CNOT(q0, q1),  # CNOT gate (q0 controls q1)
        cirq.measure(q0, q1),  # Explicit measurement key
    )
    print("Quantum Circuit:\n", circuit)
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1024)  # Run 1024 times
    print("Measurement Results:\n", result.histogram(key=[q0, q1]))
    expected_output = """Measurement Results:
    Counter({(0, 0): ~512, (1, 1): ~512})
    """


def plot_circuit_sim():
    global q0, q1, result
    # Define qubits
    q0, q1 = cirq.LineQubit.range(2)
    # Create circuit with explicit measurement keys
    circuit = cirq.Circuit(
        cirq.H(q0),  # Hadamard gate on q0
        cirq.CNOT(q0, q1),  # CNOT gate (q0 controls q1)
        cirq.measure(q0, key="m0"),  # Explicit measurement key
        cirq.measure(q1, key="m1")
    )
    # Simulate circuit
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1024)
    # Fix: Use the correct measurement key names
    counts = result.multi_measurement_histogram(keys=["m0", "m1"])
    print(f"Quantum circuit: \n {circuit}")

    # Plot results
    plt.bar([str(k) for k in counts.keys()], counts.values())
    plt.xlabel("Measurement Outcomes")
    plt.ylabel("Frequency")
    plt.show()


if __name__ == '__main__':
    print(cirq.__version__)

    # simple_circuit()
    plot_circuit_sim()