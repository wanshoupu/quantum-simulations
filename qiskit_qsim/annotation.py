from qiskit.circuit import QuantumCircuit, Gate
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # Subcircuit
    sub_qc = QuantumCircuit(2, name='my_group')
    sub_qc.h(0)
    sub_qc.cx(0, 1)
    sub_inst = sub_qc.to_instruction()

    # Main circuit
    qc = QuantumCircuit(2)
    qc.append(sub_inst, [0, 1])
    qc.measure_all()
    qc.draw('mpl')
    plt.show()
