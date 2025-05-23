from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class Qubit:
    """
    Represents a single qubit with a non-negative integer ID.
    """
    qid: int
    ancilla: bool = False

    def __post_init__(self):
        assert self.qid >= 0, f'Qubit {self.qid} must be >= 0'

    def __repr__(self):
        prefix = 'a' if self.ancilla else 'q'
        return f'{prefix}{self.qid}'

    def __str__(self):
        return repr(self)

    def __eq__(self, __value):
        if not isinstance(__value, Qubit):
            return NotImplemented(f'Comparison is undefined for type {type(__value)}')
        return self.qid == __value.qid and self.ancilla == __value.ancilla

    def __hash__(self):
        return hash(self.qid) + hash(self.ancilla)

    def __lt__(self, __value):
        if not isinstance(__value, Qubit):
            return NotImplemented(f'Comparison is undefined for type {type(__value)}')
        if self.ancilla != __value.ancilla:
            return self.ancilla < __value.ancilla
        return self.qid < __value.qid

    def __le__(self, __value):
        if not isinstance(__value, Qubit):
            return NotImplemented(f'Comparison is undefined for type {type(__value)}')
        return self < __value or self == __value

    def to_dict(self) -> Dict:
        return asdict(self)
