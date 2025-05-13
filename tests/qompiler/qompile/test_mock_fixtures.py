from tests.qompiler.qompile.mock_fixtures import mock_config


def test_mock_config(mocker):
    offset = 5
    emit = "SINGLET"
    config = mock_config(mocker, emit=emit, ancilla_offset=offset)
    assert config.target == "CIRQ"
    assert config.emit == emit
    device = config.device
    assert device.ancilla_offset == offset
