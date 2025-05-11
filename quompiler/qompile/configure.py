import json
import os
from dataclasses import dataclass, asdict
from typing import Dict

from jsonschema import validate

from quompiler.construct.types import QompilePlatform


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
class QompilerWarnings:
    all: bool = False
    as_errors: bool = False

    @staticmethod
    def from_dict(data: Dict) -> "QompilerWarnings":
        return QompilerWarnings(
            all=data.get("all", False),
            as_errors=data.get("as_errors", False)
        )

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DeviceConfig:
    """
    :param dimension:
    :param qspace: is the main space for computational qubits
    :param aspace: is the space for ancilla qubits. The two are non-overlapping
    """
    dimension: int
    qrange: list[int]
    arange: list[int]

    @staticmethod
    def from_dict(data: Dict) -> "DeviceConfig":
        return DeviceConfig(
            dimension=data.get("dimension", 0),
            qrange=data.get("arange", [0, 100]),
            arange=data.get("arange", [100, 200]),
        )

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class QompilerConfig:
    source: str
    output: str
    optimization: str
    debug: bool
    warnings: QompilerWarnings
    target: str
    device: DeviceConfig
    emit: str
    dump_ir: bool
    gates: list[str]
    rtol: float
    atol: float

    @staticmethod
    def from_dict(data: Dict) -> "QompilerConfig":
        validate_config(data)
        return QompilerConfig(
            source=data["source"],
            output=data.get("output", "a.out"),
            optimization=data.get("optimization", "O0"),
            debug=data.get("debug", False),
            warnings=QompilerWarnings.from_dict(data.get("warnings", {})),
            target=data.get("target", "CIRQ"),
            emit=data.get("emit", "SINGLET"),
            dump_ir=data.get("dump_ir", False),
            device=DeviceConfig.from_dict(data.get("device", {"dimension": 0})),
            gates=data.get("gates", "IXYZHST".split()),
            rtol=float(data.get("rtol", "1.e-5")),
            atol=float(data.get("atol", "1.e-8")),
        )

    def to_dict(self) -> Dict:
        result = asdict(self)
        result["warnings"] = self.warnings.to_dict()
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @staticmethod
    def from_file(json_file: str) -> "QompilerConfig":
        data = json.load(open(json_file))
        return QompilerConfig.from_dict(data)


def validate_config(data: Dict):
    schema_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "config_schema.json"))
    assert os.path.exists(schema_file)
    schema = json.load(open(schema_file))
    validate(instance=data, schema=schema)
