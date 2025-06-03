from quompiler.circuits.factory_manager import FactoryManager
from quompiler.config.config_manager import ConfigManager
from quompiler.config.construct import QompilerConfig


def mock_config(emit: str = "SINGLET", ancilla_offset=1, target="CIRQ", output="a.out") -> QompilerConfig:
    man = mock_config_manager(emit, ancilla_offset, target, output)
    return man.create_config()


def mock_config_manager(emit: str = "SINGLET", ancilla_offset=1, target="CIRQ", output="a.out", lookup_tol=.4) -> ConfigManager:
    config = {
        "output": output,
        "emit": emit,
        "target": target,
        "device": {
            "ancilla_offset": ancilla_offset,
        },
        "lookup_tol": lookup_tol
    }
    man = ConfigManager()
    man.merge(config)
    return man


def mock_factory_manager(emit: str = "SINGLET", ancilla_offset=1, target="CIRQ", output="a.out", lookup_tol=.4) -> FactoryManager:
    man = mock_config_manager(emit, ancilla_offset, target, output, lookup_tol)
    return FactoryManager(man)
