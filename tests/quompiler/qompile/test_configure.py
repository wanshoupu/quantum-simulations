import json
import os.path

import pytest

from quompiler.circuits.cirq_circuit import CirqBuilder
from quompiler.circuits.qiskit_circuit import QiskitBuilder
from quompiler.qompile.configure import QompilePlatform, QompilePlatformEncoder, qompile_platform_decoder, QompilerConfig, DeviceConfig, EmitType


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


def test_QompilerConfig_from_file():
    fpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "test_compiler_config.json"))
    assert os.path.exists(fpath)

    # execute
    config = QompilerConfig.from_file(fpath)

    # verify
    assert config is not None
    assert isinstance(config, QompilerConfig)
    assert config.device is not None
    assert isinstance(config.device, DeviceConfig)
    assert config.device.dimension == 8


def test_EmitType_comparison():
    lesser = EmitType.TWO_LEVEL
    larger = EmitType.UNIV_GATE
    assert lesser < larger
