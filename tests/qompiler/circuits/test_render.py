import os
from unittest.mock import MagicMock

from quompiler.circuits.render import QRenderer
from quompiler.utils.format_matrix import MatrixFormatter

formatter = MatrixFormatter(precision=2)


def test_render(mocker):
    MockBuilder = mocker.patch("quompiler.circuits.qbuilder.CircuitBuilder")
    mock_config = mocker.patch("quompiler.config.construct.QompilerConfig")
    builder = MockBuilder()
    codefile = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "codefile.qco"))
    # execute
    render = QRenderer(config=mock_config, builder=builder)
    circuit = render.render(codefile)
    # verify
    builder.build_group.assert_called_once()
    builder.build_gate.assert_called_once()
    builder.finish.assert_called_once()
    assert isinstance(circuit, MagicMock)
