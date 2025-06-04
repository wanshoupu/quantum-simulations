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

    def read_source(self) -> NDArray:
        """
        Load a unitary matrix stored in a file.
        Returns the loaded unitary matrix in memory.
        """
        return load_ndarray(self._source)

    def merge(self, data: dict) -> 'ConfigManager':
        self._config = merge_dicts(self._config, data)
        return self

    def load_config_file(self, json_file: str) -> 'ConfigManager':
        """
        Loads a configuration from a JSON file.
        :param json_file: a path to the JSON configuration file.
        """
        config = json.load(open(json_file))
        self.merge(config)
        return self

    def parse_args(self) -> 'ConfigManager':
        args, _ = self._parser.parse_known_args()
        if args.source:
            self._source = args.source
        if args.config:
            self.load_config_file(args.config)
        qconfig = self._to_config(args)
        self.merge(qconfig)
        return self

    def help(self):
        return self._parser.format_help()

    def create_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("source", nargs="?", help="The source to parse, if any")
        parser.add_argument("-c", "--config",
                            help="The config file name (JSON format). If provided, it will override the default config "
                                 "but will be overridden by other arguments")
        parser.add_argument("-o", "--output", help="output file name")
        parser.add_argument("-O", "--opt", help="optimization level", choices=["0", "1", "2", "3"])
        parser.add_argument("-g", "--debug", help="Print debug level info", action="store_true")
        parser.add_argument("-Wall", help="Enable all commonly used warning messages", action="store_true")
        parser.add_argument("-Werror", help="Whether or not treat warning as error", action="store_true")
        parser.add_argument("--target", help="Target quantum computing platform", choices=["CIRQ", "QISKIT", "QUIMB"])
        parser.add_argument("--emit",
                            help="specifies what kind of output file the compiler should generate after processing the source code.",
                            choices=["INVALID", "UNITARY", "TWO_LEVEL", "SINGLET", "MULTI_TARGET", "CTRL_PRUNED", "UNIV_GATE", "CLIFFORD_T"])
        parser.add_argument("--rtol", help="Relative tolerance — allows for proportional error")
        parser.add_argument("--atol", help="Absolute tolerance — allows for fixed error")
        parser.add_argument("--lookup_tol", help="SU2Net lookup tolerance for Solovay-Kitaev decomposition.")
        parser.add_argument("--ancilla_offset", help="The qubit space offset for ancilla qubits")
        return parser

    def save(self, file_path):
        with open(file_path, "w") as f:
            json.dump(self._config, f, indent=2)

    def _to_config(self, args):
        result = {}
        if args.source:
            result["source"] = args.source
        if args.output:
            result["output"] = args.output
        if args.opt:
            result["optimization"] = f"O{args.opt}"
        if args.debug:
            result["debug"] = args.debug
        if args.Wall:
            if "warnings" not in result:
                result["warnings"] = {}
            result["warnings"]["all"] = args.Wall
        if args.Werror:
            if "warnings" not in result:
                result["warnings"] = {}
            result["warnings"]["error"] = args.Werror
        if args.ancilla_offset:
            result["device"] = {
                "ancilla_offset": int(args.ancilla_offset),
            }
        if args.target:
            result["target"] = args.target
        if args.emit:
            result["emit"] = args.emit
        if args.rtol:
            result["rtol"] = float(args.rtol)
        if args.atol:
            result["atol"] = float(args.atol)
        return result


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict):
            assert isinstance(value, dict), f'{value} should be a dict'
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def create_config(emit: str = "SINGLET", ancilla_offset=1, target="CIRQ", output="a.out", lookup_tol=.4) -> QompilerConfig:
    man = create_config_manager(emit, ancilla_offset, target, output, lookup_tol)
    return man.create_config()


def create_config_manager(emit: str = "SINGLET", ancilla_offset=1, target="CIRQ", output="a.out", lookup_tol=.4) -> ConfigManager:
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
