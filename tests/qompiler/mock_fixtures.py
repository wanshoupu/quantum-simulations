from quompiler.circuits.factory_manager import FactoryManager
from quompiler.config.config_manager import ConfigManager
from quompiler.config.construct import QompilerConfig


def mock_device(mocker, ancilla_offset):
    device_cls = mocker.patch("quompiler.config.construct.DeviceConfig")
    result = device_cls.return_value
    result.ancilla_offset = ancilla_offset
    return result


def mock_config(mocker, emit: str = "SINGLET", ancilla_offset=1, target="CIRQ") -> QompilerConfig:
    config_cls = mocker.patch("quompiler.config.construct.QompilerConfig")

    result = config_cls.return_value
    result.device = mock_device(mocker, ancilla_offset)
    result.target = target
    result.emit = emit

    return result


def mock_factory_manager(mocker, emit: str = "SINGLET", ancilla_offset=1, target="CIRQ") -> FactoryManager:
    config = mock_config(mocker, emit, ancilla_offset, target)
    mocker.patch.object(ConfigManager, "get_config", return_value=config)
    factory_manager_cls = mocker.patch("quompiler.circuits.factory_manager.FactoryManager")
    factory_manager = factory_manager_cls.return_value
    factory_manager.config_man = ConfigManager()
    return factory_manager


def mock_decompose(mocker):
    mock_std_decompose = mocker.patch("quompiler.utils.std_decompose.std_decompose", return_value="mocked data")
    mock_ctrl_decompose = mocker.patch("quompiler.utils.std_decompose.ctrl_decompose", return_value="mocked data")
    mock_std_decompose.assert_called_once()
    mock_ctrl_decompose.assert_called_once()
