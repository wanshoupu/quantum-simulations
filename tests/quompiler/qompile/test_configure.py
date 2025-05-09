import json

import pytest

from quompiler.circuits.cirq_circuit import CirqBuilder
from quompiler.circuits.qiskit_circuit import QiskitBuilder
from quompiler.qompile.configure import QompilePlatform, QompilePlatformEncoder, qompile_platform_decoder


def test_QompilePlatform_instantiation():
    builder = QompilePlatform.CIRQ(dimension=3)
    assert builder is not None


@pytest.mark.parametrize("name, builder", [
    ['CIRQ', CirqBuilder],
    ['QISKIT', QiskitBuilder],
])
def test_QompilePlatform_get_by_name(name, builder):
    t = QompilePlatform[name]
    assert t.value == builder


@pytest.mark.parametrize("name, expected", [
    ['Cirq', QompilePlatform.CIRQ],
])
def test_QompilePlatform_serde(name, expected):
    # Serialize
    json_str = json.dumps({"platform": QompilePlatform.CIRQ}, cls=QompilePlatformEncoder)

    # Deserialize
    platform_obj = json.loads(json_str, object_hook=qompile_platform_decoder)

    # verify
    actual = platform_obj["platform"]
    assert actual == expected
