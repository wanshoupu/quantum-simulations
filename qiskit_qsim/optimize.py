from qiskit_ibm_runtime import QiskitRuntimeService


if __name__ == '__main__':
    service = QiskitRuntimeService()

    backend = service.least_busy(simulator=False, operational=True)

    # Convert to an ISA circuit and layout-mapped observables.
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(qc)

    isa_circuit.draw("mpl", idle_wires=False)