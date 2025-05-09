import argparse
import json


def cli_to_config(args):
    return {
        "source": args.source,
        "output": args.output,
        "optimization": f"O{args.opt}",
        "debug": args.debug,
        "warnings": {
            "all": args.Wall,
            "as_errors": args.Werror
        },
        "target": args.target,
        "emit": args.emit,
        "dump_ir": args.dump_ir
    }


parser = argparse.ArgumentParser()
parser.add_argument("source")
parser.add_argument("-o", "--output", help="", default="a.out")
parser.add_argument("-O", "--opt", help="", choices=["0", "1", "2", "3"], default="0")
parser.add_argument("-g", "--debug", help="", action="store_true")
parser.add_argument("-Wall", help="", action="store_true")
parser.add_argument("-Werror", help="", action="store_true")
parser.add_argument("--target", help="", choices='cirq,qiskit,pennylane,qfactor,qutip', default="x86")
parser.add_argument("--emit", help="specifies what kind of output file the compiler should generate after processing the source code.", choices=["asm", "obj", "ir"], default="obj")
parser.add_argument("--dump-ir", help="", action="store_true")

args = parser.parse_args()
config = cli_to_config(args)

with open("compiler_config.json", "w") as f:
    json.dump(config, f, indent=2)
