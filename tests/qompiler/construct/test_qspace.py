import random

import numpy as np
import pytest

from quompiler.construct.qspace import Qubit, Ancilla


def test_qubit_init_invalid():
    q0 = Qubit(0)
    assert q0 is not None
    with pytest.raises(AssertionError):
        Qubit(-50)


def test_qubit_eq():
    q0 = Qubit(0)
    q1 = Qubit(1)
    q2 = Qubit(1)
    assert q0 != q1
    assert q1 == q2


def test_qubit_hash():
    q0 = Qubit(0)
    q1 = Qubit(1)
    q2 = Qubit(1)
    assert hash(q0) != hash(q1)
    assert hash(q1) == hash(q2)


def test_qubit_set():
    qubits = {Qubit(i) for i in range(3)}
    others = {Qubit(i) for i in range(4)}
    assert qubits & others == {Qubit(i) for i in range(3)}
    assert qubits | others == {Qubit(i) for i in range(4)}


def test_qubit_dict():
    qubit_dict = {Qubit(i): i for i in range(3)}
    for i in range(3):
        qubit_dict[Qubit(i)] = i + 3
    assert qubit_dict == {Qubit(i): i + 3 for i in range(3)}


def test_ancilla_array():
    dim = 100
    qids = [Ancilla(i) for i in range(dim)]
    assert all(isinstance(q, Qubit) for q in qids)


def test_qubit_init_comparison():
    assert Qubit(1) == Qubit(1)
    assert Qubit(1) < Qubit(2)
    assert Qubit(51) >= Qubit(0)


def test_qubits_sorting():
    n = 10
    qubits = [Qubit(random.randint(0, 10)) for _ in range(n)]
    qubits.sort()
    assert all(qubits[i - 1] <= qubits[i] for i in range(1, n))
    assert all(qubits[i] >= qubits[i - 1] for i in range(1, n))

    qset = set(qubits)
    assert len(qset) < len(qubits)
