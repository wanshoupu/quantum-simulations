from quompiler.config.config_manager import create_config, create_config_manager


def test_create_config():
    offset = 5
    emit = "MULTI_TARGET"
    target = "QUIMB"
    config = create_config(emit=emit, ancilla_offset=offset, target=target)
    assert config.target == target
    assert config.emit == emit
    device = config.device
    assert device.ancilla_offset == offset


def test_create_config_manager():
    offset = 5
    emit = "SINGLET"
    target = "QISKIT"

    # execute
    man = create_config_manager(emit=emit, ancilla_offset=offset, target=target)

    # verify
    config = man.create_config()
    assert config.target == target
    assert config.emit == emit
    device = config.device
    assert device.ancilla_offset == offset
