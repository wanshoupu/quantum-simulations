from sympy.physics.quantum import TensorProduct, represent
from sympy.physics.quantum.gate import IdentityGate, XGate

if __name__ == '__main__':
    # Specify the number of qubits in the system
    x_on_qubit1 = TensorProduct(IdentityGate(1), XGate(1))
    x_matrix_2q = represent(x_on_qubit1, nqubits=2)
    print(x_matrix_2q)
