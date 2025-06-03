from quompiler.circuits.abstract_factory import QFactory
from quompiler.circuits.cirq_factory.cirq_factory import CirqFactory
from quompiler.circuits.qiskit_factory.qiskit_factory import QiskitFactory
from quompiler.circuits.quimb_factory.qumb_factory import QuimbFactory
from quompiler.config.config_manager import ConfigManager
from quompiler.construct.types import QompilePlatform


class FactoryManager:
    def __init__(self, config: ConfigManager = None):
        self.config_man = config or ConfigManager()

    def parse_args(self):
        self.config_man.parse_args()

    def create_factory(self) -> QFactory:
        config = self.config_man.create_config()
        platform = QompilePlatform[config.target]
        if platform == QompilePlatform.CIRQ:
            return CirqFactory(config)
        if platform == QompilePlatform.QISKIT:
            return QiskitFactory(config)
        if platform == QompilePlatform.QUIMB:
            return QuimbFactory(config)
        raise ValueError(f"Unsupported platform {platform}")
