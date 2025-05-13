from quompiler.circuits.factory_manager import FactoryManager
from quompiler.config.config_manager import ConfigManager
from quompiler.config.construct import QompilerConfig


def mock_config(emit: str = "SINGLET", ancilla_offset=1, target="CIRQ") -> QompilerConfig:
    man = mock_config_manager(emit, ancilla_offset, target)
    return man.create_config()


def mock_config_manager(emit: str = "SINGLET", ancilla_offset=1, target="CIRQ") -> ConfigManager:
    config = {
        "emit": emit,
        "target": target,
        "device": {
            "ancilla_offset": ancilla_offset,
        }
    }
    man = ConfigManager()
    man.load_config(config)
    return man


def mock_factory_manager(emit: str = "SINGLET", ancilla_offset=1, target="CIRQ") -> FactoryManager:
    man = mock_config_manager(emit, ancilla_offset, target)
    return FactoryManager(man)


def mock_decompose(mocker):
    mock_std_decompose = mocker.patch("quompiler.utils.std_decompose.std_decompose", return_value="mocked data")
    mock_ctrl_decompose = mocker.patch("quompiler.utils.std_decompose.ctrl_decompose", return_value="mocked data")
    mock_std_decompose.assert_called_once()
    mock_ctrl_decompose.assert_called_once()
