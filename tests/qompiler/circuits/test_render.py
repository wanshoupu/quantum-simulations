import os
import tempfile
from unittest.mock import MagicMock

from quompiler.circuits.qfactory import QFactory
from quompiler.circuits.render import QRenderer
from quompiler.config.config_manager import ConfigManager
from quompiler.utils.file_io import CODE_FILE_EXT
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_unitary

formatter = MatrixFormatter(precision=2)


def create_codefile(codefile):
    n = 3
    config = ConfigManager().merge(dict(emit='PRINCIPAL', ancilla_offset=n, optimization='O3', output=codefile)).create_config()
    factory = QFactory(config)
    dim = 1 << n
    input_mat = random_unitary(dim)
    compiler = factory.get_qompiler()
    compiler.compile(input_mat)


def test_render(mocker):
    with tempfile.NamedTemporaryFile(suffix=CODE_FILE_EXT, mode="w+b", delete=True) as codefile:
        create_codefile(codefile.name)

        MockBuilder = mocker.patch("quompiler.circuits.qbuilder.CircuitBuilder")
        builder = MockBuilder()
        # execute
        render = QRenderer(builder=builder)
        circuit = render.render(codefile.name)
        # verify
        builder.build_group.assert_called()
        builder.build_gate.assert_called()
        builder.finish.assert_called()
        assert isinstance(circuit, MagicMock)
