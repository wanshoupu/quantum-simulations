from tests.qompiler.mock_fixtures import mock_config, mock_config_manager


def test_mock_config():
    offset = 5
    emit = "MULTI_TARGET"
    target = "QUIMB"
    config = mock_config(emit=emit, ancilla_offset=offset, target=target)
    assert config.target == target
    assert config.emit == emit
    device = config.device
    assert device.ancilla_offset == offset


def test_mock_config_manager():
    offset = 5
    emit = "SINGLET"
    target = "QISKIT"

    # execute
    man = mock_config_manager(emit=emit, ancilla_offset=offset, target=target)

    # verify
    config = man.create_config()
    assert config.target == target
    assert config.emit == emit
    device = config.device
    assert device.ancilla_offset == offset
