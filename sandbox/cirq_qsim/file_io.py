import json
import os

import cirq

if __name__ == '__main__':
    # Create a simple circuit
    qubit = cirq.NamedQubit("q")
    circuit = cirq.Circuit(cirq.X(qubit) ** 0.5, cirq.measure(qubit, key='m'))
    print("Original circuit:")
    print(circuit)

    # Serialize to file
    j = cirq.to_json(circuit)
    print(j)

    # Write to file
    filename = 'my_circuit.json'
    with open(filename, 'w') as f:
        json.dump(j, f, indent=4)

    # Deserialize from file
    loaded_circuit = cirq.read_json(filename)
    print("\nLoaded circuit:")
    print(loaded_circuit)

    # cleanup
    os.remove(filename)
