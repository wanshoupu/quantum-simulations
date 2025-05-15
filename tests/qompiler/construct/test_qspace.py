import random

import numpy as np
import pytest

from quompiler.construct.qspace import QSpace, Qubit, Ancilla


def test_qspace_init_random():
    dim = 100
    for _ in range(10):
        qids = np.random.choice(dim, size=10, replace=False)
        qs = QSpace(qids)
        assert qs
        sorting = np.argsort(qids).tolist()
        # print(sorting)
        assert sorting == qs.sorting


def test_qubit_init_invalid():
    q0 = Qubit(0)
    assert q0 is not None
    with pytest.raises(AssertionError):
        Qubit(-50)


def test_init_int_array():
    dim = 100
    qids = np.random.choice(dim, size=100, replace=False)
    qs = QSpace(qids)
    assert all(isinstance(q, Qubit) for q in qs)


def test_init_ancilla_array():
    dim = 100
    qids = [Ancilla(i) for i in range(dim)]
    qs = QSpace(qids)
    assert all(isinstance(q, Qubit) for q in qs)


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
