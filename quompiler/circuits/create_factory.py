from quompiler.circuits.abstract_factory import QFactory
from quompiler.circuits.cirq_factory.cirq_factory import CirqFactory
from quompiler.circuits.qiskit_factory.qiskit_factory import QiskitFactory
from quompiler.circuits.quimb_factory.qumb_factory import QuimbFactory
from quompiler.config.construct import QompilerConfig
from quompiler.construct.types import QompilePlatform


def create_factory(config: QompilerConfig) -> QFactory:
    platform = QompilePlatform[config.target]
    if platform == QompilePlatform.CIRQ:
        return CirqFactory(config)
    if platform == QompilePlatform.QISKIT:
        return QiskitFactory(config)
    if platform == QompilePlatform.QUIMB:
        return QuimbFactory(config)
    raise ValueError(f"Unsupported platform {platform}")
