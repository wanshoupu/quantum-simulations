from quompiler.circuits.qbuilder import CircuitBuilder
from quompiler.config.construct import QompilerConfig
from quompiler.construct.bytecode import BytecodeRevIter
from quompiler.optimize.code_analyze import gen_stats, collect_qspace
from quompiler.utils.file_io import read_code


class QRenderer:

    def __init__(self, config: QompilerConfig, builder: CircuitBuilder):
        self.config = config
        self.builder = builder

    def render(self, codefile):
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
