from quompiler.circuits.factory_manager import FactoryManager
from quompiler.config.config_manager import ConfigManager
from quompiler.config.construct import QompilerConfig


def mock_config(emit: str = "SINGLET", ancilla_offset=1, target="CIRQ", output="a.out") -> QompilerConfig:
    man = mock_config_manager(emit, ancilla_offset, target, output)
    return man.create_config()


def mock_config_manager(emit: str = "SINGLET", ancilla_offset=1, target="CIRQ", output="a.out") -> ConfigManager:
    config = {
        "output": output,
        "emit": emit,
        "target": target,
        "device": {
            "ancilla_offset": ancilla_offset,
        }
    }
    man = ConfigManager()
    man.load_config(config)
    return man


def mock_factory_manager(emit: str = "SINGLET", ancilla_offset=1, target="CIRQ", output="a.out") -> FactoryManager:
    man = mock_config_manager(emit, ancilla_offset, target, output)
    return FactoryManager(man)
