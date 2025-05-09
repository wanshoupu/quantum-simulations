from dataclasses import dataclass, field, asdict
from typing import Dict
import json


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
class CompilerConfig:
    source: str
    output: str = "a.out"
    optimization: str = "O0"
    debug: bool = False
    warnings: CompilerWarnings = field(default_factory=CompilerWarnings)
    target: str = "x86"
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
            target=data.get("target", "x86"),
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
