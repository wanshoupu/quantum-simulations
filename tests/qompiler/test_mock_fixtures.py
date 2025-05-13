from tests.qompiler.mock_fixtures import mock_config, mock_factory_manager, mock_config_manager


def test_mock_config(mocker):
    offset = 5
    emit = "MULTI_TARGET"
    target = "QUIMB"
    config = mock_config(emit=emit, ancilla_offset=offset, target=target)
    assert config.target == target
    assert config.emit == emit
    device = config.device
    assert device.ancilla_offset == offset


def test_mock_config_manager(mocker):
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


def test_mock_factory(mocker):
    emit_type = "CLIFFORD_T"
    offset = 51
    target_platform = "CIRQ"
    fman = mock_factory_manager(emit=emit_type, ancilla_offset=offset, target=target_platform)
    assert fman is not None
    factory = fman.create_factory()
    assert factory is not None
    qompiler = factory.get_qompiler()
    assert qompiler is not None
    qdevice = factory.get_device()
    assert qdevice is not None
    device = qompiler.device
    assert device.aoffset == offset
