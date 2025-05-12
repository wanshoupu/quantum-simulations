from tests.quompiler.qompile.mock_fixtures import mock_config


def test_mock_config(mocker):
    aspace = 5
    emit = "SINGLET"
    config = mock_config(mocker, emit=emit, aspace=aspace)
    assert config.target == "CIRQ"
    assert config.emit == emit
    device = config.device
    assert device.arange[0] == aspace
