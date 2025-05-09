from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Type
import json

from quompiler.circuits.circuit_builder import CircuitBuilder
from quompiler.circuits.cirq_circuit import CirqBuilder
from quompiler.circuits.qiskit_circuit import QiskitBuilder
from quompiler.circuits.quimb_circuit import QuimbBuilder


class QompilePlatform(Enum):
    CIRQ = CirqBuilder
    QISKIT = QiskitBuilder
    QUIMB = QuimbBuilder

    def __call__(self, dimension: int) -> CircuitBuilder:
        return self.value(dimension)


class QompilePlatformEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, QompilePlatform):
            return obj.name
        return super().default(obj)


def qompile_platform_decoder(dct: dict):
    if "platform" in dct:
        dct["platform"] = QompilePlatform[dct["platform"]]
    return dct


@dataclass
class CompilerWarnings:
    all: bool = False
    as_errors: bool = False

    @staticmethod
    def from_dict(data: Dict) -> "CompilerWarnings":
        return CompilerWarnings(
            all=data.get("all", False),
            as_errors=data.get("as_errors", False)
        )

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DeviceConfig:
    dimension: int

    @staticmethod
    def from_dict(data: Dict) -> "DeviceConfig":
        return DeviceConfig(
            dimension=data.get("dimension", 0),
        )

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CompilerConfig:
    source: str
    output: str = "a.out"
    optimization: str = "O0"
    debug: bool = False
    warnings: CompilerWarnings = field(default_factory=CompilerWarnings)
    target: str = "CIRQ"
    device: DeviceConfig = field(default_factory=DeviceConfig)
    emit: str = "obj"
    dump_ir: bool = False

    @staticmethod
    def from_dict(data: Dict) -> "CompilerConfig":
        return CompilerConfig(
            source=data["source"],
            output=data.get("output", "a.out"),
            optimization=data.get("optimization", "O0"),
            debug=data.get("debug", False),
            warnings=CompilerWarnings.from_dict(data.get("warnings", {})),
            target=data.get("target", "CIRQ"),
            emit=data.get("emit", "obj"),
            dump_ir=data.get("dump_ir", False)
        )

    def to_dict(self) -> Dict:
        result = asdict(self)
        result["warnings"] = self.warnings.to_dict()
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @staticmethod
    def from_json(json_str: str) -> "CompilerConfig":
        return CompilerConfig.from_dict(json.loads(json_str))
