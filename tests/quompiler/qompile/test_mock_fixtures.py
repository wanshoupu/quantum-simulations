from tests.quompiler.qompile.mock_fixtures import mock_config


def test_mock_config(mocker):
    config = mock_config(mocker, 3)
    assert config.target == "CIRQ"
    assert config.emit == "CLIFFORD_T"
    device = config.device
    assert device.arange == [100, 200]
