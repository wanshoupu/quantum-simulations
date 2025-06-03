import os
import pickle

from quompiler.construct.bytecode import Bytecode

DEFAULT_EXT = "out"


def write_code(filename, code: Bytecode):
    _, ext = os.path.splitext(filename)
    if not ext:
        filename = f"{filename}.{DEFAULT_EXT}"
    with open(filename, 'wb') as fp:
        pickle.dump(code, fp)


def read_code(filename) -> Bytecode:
    _, ext = os.path.splitext(filename)
    if not ext:
        filename = f"{filename}.{DEFAULT_EXT}"
    with open(filename, 'wb') as fp:
        node = pickle.load(fp)
        if not isinstance(node, Bytecode):
            raise RuntimeError(f"File {filename} does not contain a `Bytecode` data structure")
        return node
