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
            all=data["all"],
            as_errors=data["as_errors"]
        )

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DeviceConfig:
    """
    :param ancilla_offset: is the space offset for ancilla qubits so that
    computational qubits are in the range [0, offset] and ancilla qubits are [offset, inf].
    """
    ancilla_offset: int

    @staticmethod
    def from_dict(data: Dict) -> "DeviceConfig":
        return DeviceConfig(
            ancilla_offset=data["ancilla_offset"],
        )

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class QompilerConfig:
    """
    :param source: The source to parse, if any
    :param output: output file name
    :param optimization: optimization level. 0=no optimization, 1=basic optimization, 2=moderate optimization, 3=advanced optimization
    :param debug: Print debug level info if true
    :param warnings: compiler warnings settings
    :param target: target quantum computing platform
    :param device: quantum computing device setting
    :param emit: specifies the output building blocks the compiler should generate after processing the source
    :param rtol: relative tolerance allows for proportional error
    :param atol: absolute tolerance allows for fixed error
    :param lookup_tol: SU2Net lookup tolerance for Solovay-Kitaev decomposition
    """
    source: str
    output: str
    optimization: str
    debug: bool
    warnings: QompilerWarnings
    target: str
    device: DeviceConfig
    emit: str
    rtol: float
    atol: float
    lookup_tol: float

    @staticmethod
    def from_dict(data: Dict) -> "QompilerConfig":
        validate_config(data)
        device = DeviceConfig.from_dict(data["device"])
        warnings = QompilerWarnings.from_dict(data["warnings"])
        return QompilerConfig(
            source=data["source"],
            output=data["output"],
            optimization=data["optimization"],
            debug=data["debug"],
            warnings=warnings,
            target=data["target"],
            emit=data["emit"],
            device=device,
            rtol=float(data["rtol"]),
            atol=float(data["atol"]),
            lookup_tol=float(data["lookup_tol"]),
        )

    def to_dict(self) -> Dict:
        result = asdict(self)
        result["warnings"] = self.warnings.to_dict()
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def validate_config(data: Dict):
    schema_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "config_schema.json"))
    assert os.path.exists(schema_file)
    schema = json.load(open(schema_file))
    validate(instance=data, schema=schema)
