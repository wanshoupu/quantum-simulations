import os
import sys

import pytest
from jsonschema import ValidationError

from quompiler.config.config_manager import ConfigManager, merge_dicts
from quompiler.construct.types import EmitType, OptLevel, QompilePlatform

fpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "test_compiler_config.json"))


def test_parse_default_config():
    parser = ConfigManager()
    config = parser.create_config()
    assert config.device is not None
    assert config.target == "CIRQ"
    assert config.emit in {e.name for e in list(EmitType)}


@pytest.mark.parametrize("dict1,dict2,expected", [
    [{"a": 1}, {"b": 2}, {"a": 1, "b": 2}],
    [{"foo": "bar"}, {"foo": "bar"}, {"foo": "bar"}],
    [{"a": 1, "b": {"d": 2}}, {"b": {"c": 1}}, {"a": 1, "b": {"d": 2, "c": 1}}],
])
def test_merge_dicts(dict1: dict, dict2: dict, expected: dict):
    actual = merge_dicts(dict1, dict2)
    assert actual == expected


def test_parse_args(monkeypatch):
    rtol = 1e6
    atol = 1e9
    offset = 77
    target = 'QUIMB'
    test_args = ['pyfile.py', 'source_file.txt', '--target', target, "--rtol", str(rtol), "--atol", str(atol), "--ancilla_offset", str(offset)]
    monkeypatch.setattr(sys, "argv", test_args)
    config_man = ConfigManager()

    # execute
    config_man.parse_args()
    actual = config_man.create_config()

    # verify
    assert actual.target == target
    assert actual.rtol == rtol
    assert actual.atol == atol
    assert actual.device.ancilla_offset == offset


def test_save(mocker):
    test_args = ['--emit', 'UNIV_GATE', "--rtol", "1e-6", "--atol", "1e-9", "--ancilla_offset", "7"]
    mocker.patch('sys.argv', test_args)
    config_man = ConfigManager()
    config_man.parse_args()
    assert config_man._config

    mock_open = mocker.mock_open()
    mocker.patch("quompiler.config.config_manager.open", mock_open)
    mocker.patch("builtins.open", mock_open)
    file_path = "tmp_config.json"

    # execute
    config_man.save(file_path)

    # verify
    mock_open.assert_called_with(file_path, "w")
    assert mock_open().write.call_count > 0


def test_load_config_file():
    man = ConfigManager()
    default = man.create_config()

    # execute
    man.load_config_file(fpath)
    config = man.create_config()

    # verify
    assert config.output != default.output  # overwritten
    assert config.atol != default.atol  # overwritten
    assert config.target == default.target  # default left alone
    assert config.emit == default.emit  # default left alone


def test_load_config():
    man = ConfigManager()
    default = man.create_config()
    data = {
        "emit": "TWO_LEVEL",
    }
    # execute
    man.load_config(data)

    config = man.create_config()
    assert "TWO_LEVEL" == config.emit != default.emit


def test_enum_invalid_emit():
    man = ConfigManager()
    data = {
        "emit": "THIS_DOESNT_EXIST",
    }
    # execute
    with pytest.raises(ValidationError):
        man.load_config(data)
        man.create_config()


def test_in_sync_enum_emit():
    man = ConfigManager()
    for emit in EmitType:
        conf_value = emit.name
        data = {
            "emit": conf_value,
        }
        man.load_config(data)

        # execute
        config = man.create_config()
        assert conf_value == config.emit


def test_enum_invalid_optimization():
    man = ConfigManager()
    data = {
        "optimization": "THIS_DOESNT_EXIST",
    }
    # execute
    with pytest.raises(ValidationError):
        man.load_config(data)
        man.create_config()


def test_in_sync_enum_optimization():
    man = ConfigManager()
    for opt in OptLevel:
        conf_value = opt.name
        data = {
            "optimization": conf_value,
        }
        man.load_config(data)

        # execute
        config = man.create_config()
        assert conf_value == config.optimization, f'{conf_value} != {config.optimization}'


def test_enum_invalid_target():
    man = ConfigManager()
    data = {
        "target": "THIS_DOESNT_EXIST",
    }
    # execute
    with pytest.raises(ValidationError):
        man.load_config(data)
        man.create_config()


def test_in_sync_enum_target():
    man = ConfigManager()
    for opt in QompilePlatform:
        conf_value = opt.name
        data = {
            "target": conf_value,
        }
        man.load_config(data)

        # execute
        config = man.create_config()
        assert conf_value == config.target, f'{conf_value} != {config.target}'


def test_file_override_default():
    # 1 default
    man = ConfigManager()
    default = man.create_config()

    # 2 json file config
    man.load_config_file(fpath)
    final_config = man.create_config()

    # verify
    assert final_config.target == default.target  # default left alone
    assert default.output != final_config.output  # overwritten
    assert default.atol != final_config.atol  # overwritten
    assert default.device.ancilla_offset != final_config.device.ancilla_offset  # overwritten
    assert final_config.output == "program.out"
    assert final_config.atol == 1e-5
    assert final_config.device.ancilla_offset == 1000


def test_cmd_override_file(mocker):
    # 1 default
    man = ConfigManager()
    # 2 json file config
    man.load_config_file(fpath)
    file_config = man.create_config()

    # 3 cmd line args
    test_args = ["script_name.py", "-o", "bar.out", "--atol", "1e-9", "--ancilla_offset", "10000"]
    mocker.patch('sys.argv', test_args)
    man.parse_args()
    final_config = man.create_config()

    # verify
    assert file_config.source == final_config.source  # default left alone
    assert file_config.output != final_config.output  # overwritten
    assert file_config.atol != final_config.atol  # overwritten
    assert file_config.device.ancilla_offset != final_config.device.ancilla_offset  # overwritten
    assert final_config.output == "bar.out"
    assert final_config.atol == 1e-9
    assert final_config.device.ancilla_offset == 10000
