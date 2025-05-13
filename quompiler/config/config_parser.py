import argparse
import json
import os
import sys

from quompiler.config.construct import validate_config, QompilerConfig


class ConfigManager:
    """
    Manages the parsing, merging, and updating of configurations
    """

    def __init__(self):
        self.default_config_file = os.path.join(os.path.dirname(__file__), "data", "default_config.json")
        self.config = json.load(open(self.default_config_file))

    def get_config(self) -> QompilerConfig:
        validate_config(self.config)
        return QompilerConfig.from_dict(self.config)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("source")
        parser.add_argument("-o", "--output", help="output file name", default="a.out")
        parser.add_argument("-O", "--opt", help="optimization level", choices=["0", "1", "2", "3"], default="0")
        parser.add_argument("-g", "--debug", help="", action="store_true")
        parser.add_argument("-Wall", help="", action="store_true")
        parser.add_argument("-Werror", help="Whether or not treat warning as error", action="store_true")
        parser.add_argument("--target", help="Target quantum computing platform", choices=["CIRQ", "QISKIT", "QUIMB"], default="CIRQ")
        parser.add_argument("--emit",
                            help="specifies what kind of output file the compiler should generate after processing the source code.",
                            choices=["INVALID", "UNITARY", "TWO_LEVEL", "SINGLET", "MULTI_TARGET", "CTRL_PRUNED", "UNIV_GATE", "CLIFFORD_T"],
                            default="SINGLET")
        parser.add_argument("--rtol", help="specify relative tolerance", default="1e-6")
        parser.add_argument("--atol", help="specify absolute tolerance", default="1e-9")
        parser.add_argument("--ancilla_offset", help="is the space offset for ancilla qubits so that", default=100)
        parser.add_argument("--dump-ir", help="", action="store_true")

        args = parser.parse_args()
        config = self._to_config(args)
        merged = merge_dicts(self.config, config)
        self.config = merged

    def save(self, file_path):
        with open(file_path, "w") as f:
            json.dump(self.config, f, indent=2)

    def _to_config(self, args):
        return {
            "source": args.source,
            "output": args.output,
            "optimization": f"O{args.opt}",
            "debug": args.debug,
            "warnings": {
                "all": args.Wall,
                "as_errors": args.Werror
            },
            "device": {
                "ancilla_offset": int(args.ancilla_offset),
            },
            "target": args.target,
            "emit": args.emit,
            "rtol": float(args.rtol),
            "atol": float(args.atol),
            "dump_ir": args.dump_ir
        }


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict):
            assert isinstance(value, dict)
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
