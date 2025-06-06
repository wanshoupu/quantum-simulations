import os
import pickle

from quompiler.construct.bytecode import Bytecode

CODE_FILE_EXT = ".qco"


def write_code(filename, code: Bytecode):
    filename = rectify_filename(filename)
    with open(filename, 'wb') as fp:
        pickle.dump(code, fp)


def read_code(filename) -> Bytecode:
    filename = rectify_filename(filename)
    with open(filename, 'rb') as fp:
        node = pickle.load(fp)
        if not isinstance(node, Bytecode):
            raise RuntimeError(f"File {filename} does not contain a `Bytecode` data structure")
        return node


def rectify_filename(filename):
    _, ext = os.path.splitext(filename)
    if ext != CODE_FILE_EXT:
        filename = f"{filename}{CODE_FILE_EXT}"
    return filename
