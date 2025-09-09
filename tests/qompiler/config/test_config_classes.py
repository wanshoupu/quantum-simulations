import json

import pytest

from quompiler.config.construct import QompilePlatformEncoder, qompile_platform_decoder, QompilerConfig, QompilerWarnings, DeviceConfig
from quompiler.construct.types import QompilePlatform, GateGrain, OptLevel


@pytest.mark.parametrize("name, expected", [
    ['target', QompilePlatform.CIRQ],
    ['emit', GateGrain.CTRL_PRUNED],
    ['optimization', OptLevel.O1],
])
def test_QompilePlatform_serde(name, expected):
    # Serialize
    json_str = json.dumps({name: expected}, cls=QompilePlatformEncoder)

    # Deserialize
    json_obj = json.loads(json_str, object_hook=qompile_platform_decoder)

    # verify
    actual = json_obj[name]
    assert actual == expected


def test_to_json():
    config = QompilerConfig(
        output='pyfile.py',
        source='source_file.txt',
        target=QompilePlatform.CIRQ,
        rtol=1e-6,
        atol=1e-9,
        emit=GateGrain.UNIV_GATE,
        warnings=QompilerWarnings(all=True, as_errors=False),
        device=DeviceConfig(ancilla_offset=7),
        lookup_tol=1e-9,
        optimization=OptLevel.O0,
        debug=False)

    # verify
    config_str = config.to_json()
    expected = ('{"source": "source_file.txt", "output": "pyfile.py", "optimization": 0, "debug": false, "warnings": '
                '{"all": true, "as_errors": false}, "target": "CIRQ", "device": {"ancilla_offset": 7}, "emit": 192, '
                '"rtol": 1e-06, "atol": 1e-09, "lookup_tol": 1e-09}')
    assert config_str == expected


def test_from_json():
    json_str = ('{"source": "source_file.txt", "output": "pyfile.py", "optimization": 0, "debug": false, "warnings": '
                '{"all": true, "as_errors": false}, "target": "CIRQ", "device": {"ancilla_offset": 7}, "emit": 192, '
                '"rtol": 1e-06, "atol": 1e-09, "lookup_tol": 1e-09}')
    actual = json.loads(json_str, object_hook=qompile_platform_decoder)
    expected = QompilerConfig(
        output='pyfile.py',
        source='source_file.txt',
        target=QompilePlatform.CIRQ,
        rtol=1e-6,
        atol=1e-9,
        emit=GateGrain.UNIV_GATE,
        warnings=QompilerWarnings(all=True, as_errors=False),
        device=DeviceConfig(ancilla_offset=7),
        lookup_tol=1e-9,
        optimization=OptLevel.O0,
        debug=False).to_dict()

    # verify
    assert actual == expected


def test_from_dict():
    emit_type = "UNIV_GATE"
    json_dict = {"source": "source_file.txt", "output": "pyfile.py", "optimization": "O0", "debug": False, "warnings":
        {"all": True, "as_errors": False}, "target": "CIRQ", "device": {"ancilla_offset": 7}, "emit": emit_type,
                 "rtol": 1e-06, "atol": 1e-09, "lookup_tol": 1e-09}
    actual = QompilerConfig.from_dict(json_dict)
    expected = QompilerConfig(
        output='pyfile.py',
        source='source_file.txt',
        target=QompilePlatform.CIRQ,
        rtol=1e-6,
        atol=1e-9,
        emit=GateGrain.UNIV_GATE,
        warnings=QompilerWarnings(all=True, as_errors=False),
        device=DeviceConfig(ancilla_offset=7),
        lookup_tol=1e-9,
        optimization=OptLevel.O0,
        debug=False)

    # verify
    assert actual == expected
