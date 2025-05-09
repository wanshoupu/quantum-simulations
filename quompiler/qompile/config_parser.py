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
parser.add_argument("-o", "--output", default="a.out")
parser.add_argument("-O", "--opt", choices=["0", "1", "2", "3"], default="0")
parser.add_argument("-g", "--debug", action="store_true")
parser.add_argument("-Wall", action="store_true")
parser.add_argument("-Werror", action="store_true")
parser.add_argument("--target", choices=["x86", "arm", "wasm"], default="x86")
parser.add_argument("--emit", choices=["asm", "obj", "ir"], default="obj")
parser.add_argument("--dump-ir", action="store_true")

args = parser.parse_args()
config = cli_to_config(args)

with open("compiler_config.json", "w") as f:
    json.dump(config, f, indent=2)
