import sympy
import cirq
import matplotlib.pyplot as plt

if __name__ == '__main__':
    q0, q1 = cirq.LineQubit.range(2)
    theta = sympy.Symbol("theta")  # Define a symbolic variable
    param_circuit = cirq.Circuit(
        cirq.rx(theta)(q0),  # Rotation around X-axis with symbolic theta
        cirq.measure(q0, key="m0")
    )
    print("Parameterized Circuit:\n", param_circuit)

    resolver = cirq.ParamResolver({"theta": 3.14 / 2})  # Set theta = π/2

    simulator = cirq.Simulator()
    result = simulator.run(param_circuit, repetitions=1000, param_resolver=resolver)
    print("Parameterized Circuit Measurement:", result.histogram(key="m0"))

    # Extract multi-qubit measurement results
    counts = result.multi_measurement_histogram(keys=["m0", "m1"])

    # Convert to a readable format
    formatted_counts = {"".join(map(str, k)): v for k, v in counts.items()}
    print("Formatted Results:", formatted_counts)

    # Plot results
    plt.bar(formatted_counts.keys(), formatted_counts.values())
    plt.xlabel("Measurement Outcomes")
    plt.ylabel("Frequency")
    plt.title("Quantum Circuit Measurement Results")
    plt.show()
