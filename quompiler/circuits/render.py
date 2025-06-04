from quompiler.circuits.qbuilder import CircuitBuilder
from quompiler.construct.bytecode import BytecodeRevIter
from quompiler.optimize.code_analyze import gen_stats, collect_qspace
from quompiler.utils.file_io import read_code


class QRenderer:

    def __init__(self, builder: CircuitBuilder):
        self.builder = builder

    def render(self, codefile) -> object:
        """
        Render codefile as quantum circuit on a target platform (depending on the type of builders used).
        Render program first calls builder to register the needed qspace (qubits, ancillas, etc.);
        Then it starts to build the gates iteratively.

        :param codefile: the file that stores the compiled code for the quantum circuit.
        :return: the finished quantum circuit specific to the target platform (depending on the type of builders used).
        """
        code = read_code(codefile)
        stats = gen_stats(code)
        qspace = collect_qspace(stats)
        self.builder.register(qspace)
        for c in BytecodeRevIter(code):
            m = c.data
            if not c.is_leaf():
                self.builder.build_group(m)
            else:
                self.builder.build_gate(m)
        return self.builder.finish()
