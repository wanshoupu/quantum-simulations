import cirq
import matplotlib.pyplot as plt
import numpy as np
import sympy
from scipy.optimize import minimize

# Create qubits
q0, q1 = cirq.LineQubit.range(2)

# Define symbolic parameters
theta = sympy.Symbol('theta')
gamma = sympy.Symbol('gamma')

# Define the parameterized quantum circuit (QAOA)
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.H(q1),
    cirq.ZPowGate(exponent=gamma).on(q0),
    cirq.ZPowGate(exponent=gamma).on(q1),
    cirq.CNOT(q0, q1),
    cirq.rx(2 * theta).on(q0),
    cirq.rx(2 * theta).on(q1),
    cirq.measure(q0, key="m0"),  # Measurement key for q0
    cirq.measure(q1, key="m1")  # Measurement key for q1
)

# Print the circuit
print("Quantum Circuit:\n", circuit)


# Cost function
def cost_function(params):
    theta_value, gamma_value = params
    resolver = cirq.ParamResolver({'theta': theta_value, 'gamma': gamma_value})
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1000, param_resolver=resolver)

    # Fix: Use the correct measurement key "m0" instead of 'q0'
    counts = result.histogram(key='m0')  # Use the correct key
    return -counts.get(0, 0)  # Minimize the inverse of |00⟩ probability


if __name__ == '__main__':
    # Classical optimization using scipy
    initial_guess = [0.1, 0.1]
    result = minimize(cost_function, initial_guess, method='COBYLA')

    # Print the optimal parameters
    optimal_theta, optimal_gamma = result.x
    print(f"Optimal Theta: {optimal_theta}, Optimal Gamma: {optimal_gamma}")

    # Visualizing the cost function
    theta_vals = np.linspace(0, np.pi, 50)
    gamma_vals = np.linspace(0, np.pi, 50)
    cost_vals = np.zeros((len(theta_vals), len(gamma_vals)))

    # Evaluate cost for each combination of theta and gamma
    for i, theta_value in enumerate(theta_vals):
        for j, gamma_value in enumerate(gamma_vals):
            cost_vals[i, j] = cost_function([theta_value, gamma_value])

    # Plotting the cost function surface
    X, Y = np.meshgrid(theta_vals, gamma_vals)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, cost_vals, cmap='viridis')

    ax.set_xlabel('Theta')
    ax.set_ylabel('Gamma')
    ax.set_zlabel('Cost Function')
    plt.show()
