import json
import os
from dataclasses import dataclass, field, asdict
from enum import Enum, IntFlag, IntEnum
from typing import Dict

from jsonschema import validate, ValidationError

from quompiler.circuits.circuit_builder import CircuitBuilder
from quompiler.circuits.cirq_circuit import CirqBuilder
from quompiler.circuits.qiskit_circuit import QiskitBuilder
from quompiler.circuits.quimb_circuit import QuimbBuilder
from quompiler.construct.qspace import Ancilla


class EmitType(IntEnum):
    UNITARY = 1  # any n-order unitary matrix
    TWO_LEVEL = 2  # 2-level unitary matrix + control sequence
    SINGLET = 3,  # unitary matrix on a single target qubit + control sequence
    TOFFOLI = 4,  # no more than two qubits in the control sequence
    ONE_CTRL = 5,  # no more than a single qubit in the control sequence
    UNIV_GATE = 6,  # any of the one-letter universal gates
    CLIFFORD_T = 7,  # Clifford + T gates only


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
    dimension: int
    aspace: list[Ancilla]

    @staticmethod
    def from_dict(data: Dict) -> "DeviceConfig":
        # qspace is the main space for computational qubits
        qspace = data.get("qspace", 100)
        # aspace is the space for ancilla qubits. The two are non-overlapping
        aspace = data.get("aspace", qspace + 100)
        return DeviceConfig(
            dimension=data.get("dimension", 0),
            aspace=[Ancilla(i) for i in range(aspace)],
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
    rtol: float
    atol: float

    @staticmethod
    def from_dict(data: Dict) -> "QompilerConfig":
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
        validate_config(data)
        return QompilerConfig.from_dict(data)


def validate_config(config_data: Dict):
    schema_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "config_schema.json"))
    assert os.path.exists(schema_file)
    schema = json.load(open(schema_file))
    try:
        validate(instance=config_data, schema=schema)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e.message}")
