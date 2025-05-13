import argparse
import json
import os

from numpy.typing import NDArray

from quompiler.config.construct import QompilerConfig, validate_config
from quompiler.config.numpy_io import load_ndarray


class ConfigManager:
    """
    Manages the parsing, merging, and updating of configurations.
    ConfigManager pulls configuration information from three sources:
    1. Default configurations in config/default_config.json
    2. Custom configurations user provides through a JSON file
    3. Custom configurations user provides through command line argument
    """

    def __init__(self):
        self._default_config_file = os.path.join(os.path.dirname(__file__), "data", "default_config.json")
        self._parser = self.create_parser()
        self._config = json.load(open(self._default_config_file))
        self._source = None

    def create_config(self) -> QompilerConfig:
        validate_config(self._config)
        return QompilerConfig.from_dict(self._config)

    def has_source(self, source: str) -> bool:
        return bool(source)

    def load_source(self) -> NDArray:
        return load_ndarray(self._source)

    def load_config(self, data: dict):
        self._config = merge_dicts(self._config, data)

    def load_config_file(self, json_file: str):
        """
        Loads a configuration from a JSON file.
        :param json_file: a path to the JSON configuration file.
        """
        config = json.load(open(json_file))
        self.load_config(config)

    def parse_args(self):
        args = self._parser.parse_args()
        if args.source:
            self._source = args.source
        if args.config:
            self.load_config_file(args.config)
        qconfig = self._to_config(args)
        merged = merge_dicts(self._config, qconfig)
        self._config = merged

    def help(self):
        return self._parser.format_help()

    def create_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("source", nargs="?", help="The source to parse, if any", default="")
        parser.add_argument("-c", "--config",
                            help="The config file name (JSON format). If provided, it will override the default config "
                                 "but will be overridden by other arguments", default="")
        parser.add_argument("-o", "--output", help="output file name", default="a.out")
        parser.add_argument("-O", "--opt", help="optimization level", choices=["0", "1", "2", "3"], default="0")
        parser.add_argument("-g", "--debug", help="Print debug level info", action="store_true")
        parser.add_argument("-Wall", help="Enable all commonly used warning messages", action="store_true")
        parser.add_argument("-Werror", help="Whether or not treat warning as error", action="store_true")
        parser.add_argument("--target", help="Target quantum computing platform", choices=["CIRQ", "QISKIT", "QUIMB"], default="CIRQ")
        parser.add_argument("--emit",
                            help="specifies what kind of output file the compiler should generate after processing the source code.",
                            choices=["INVALID", "UNITARY", "TWO_LEVEL", "SINGLET", "MULTI_TARGET", "CTRL_PRUNED", "UNIV_GATE", "CLIFFORD_T"],
                            default="SINGLET")
        parser.add_argument("--rtol", help="Relative tolerance — allows for proportional error", default="1e-6")
        parser.add_argument("--atol", help="Absolute tolerance — allows for fixed error", default="1e-9")
        parser.add_argument("--ancilla_offset", help="The qubit space offset for ancilla qubits", default=100)
        return parser

    def save(self, file_path):
        with open(file_path, "w") as f:
            json.dump(self._config, f, indent=2)

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
        }


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict):
            assert isinstance(value, dict), f'{value} should be a dict'
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
