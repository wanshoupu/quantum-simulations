from quompiler.circuits.qfactory import QFactory
from quompiler.config.config_manager import ConfigManager
from quompiler.construct.types import QompilePlatform
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import cyclic_matrix

if __name__ == '__main__':
    formatter = MatrixFormatter(precision=2)
    n = 3
    dim = 1 << n
    u = cyclic_matrix(dim, 1)
    print(formatter.tostr(u))
    config_man = ConfigManager().parse_args()
    factory = QFactory(config_man.create_config())
    compiler = factory.get_qompiler()
    render = factory.get_render(QompilePlatform.CIRQ)

    # execute
    compiler.compile(u)

    codefile = factory.get_config().output
    circuit = render.render(codefile)
    print(circuit)
