import os

from quompiler.qompile.configure import DeviceConfig, QompilerConfig
from quompiler.qompile.quompiler import CircuitInterp
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import cyclic_matrix

if __name__ == '__main__':
    formatter = MatrixFormatter(precision=2)
    n = 3
    dim = 1 << n
    u = cyclic_matrix(dim, 1)
    print(formatter.tostr(u))
    cfile = os.path.abspath(os.path.join(os.path.dirname(__file__), "compiler_config.json"))
    config = QompilerConfig.from_file(cfile)
    interp = CircuitInterp(config)

    # execute
    interp.interpret(u)
    circuit = interp.finish()
    print(circuit)
