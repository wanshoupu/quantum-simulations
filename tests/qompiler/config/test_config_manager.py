import os
import sys

import pytest

from quompiler.config.config_manager import ConfigManager, merge_dicts

fpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "test_compiler_config.json"))


def test_parse_default_config():
    parser = ConfigManager()
    config = parser.create_config()
    assert config.device is not None
    assert config.target == "CIRQ"
    assert config.emit == "CTRL_PRUNED"


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
    assert config.target != default.target
    assert config.emit != default.emit


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


def test_override_order(mocker):
    # 1 default
    man = ConfigManager()
    default = man.create_config()

    # 2 json file config
    man.load_config_file(fpath)
    file_config = man.create_config()

    # 3 cmd line args
    test_args = ['--emit', 'UNIV_GATE', "--rtol", "1e-6", "--atol", "1e-9", "--ancilla_offset", "10000"]
    mocker.patch('sys.argv', test_args)
    man.parse_args()
    cmd_config = man.create_config()

    # verify
    default_device = default.device
    file_device = file_config.device
    cmd_device = cmd_config.device
    # print('default_device\n', default_device)
    # print('file_device\n', file_device)
    # print('cmd_device\n', cmd_device)

    assert default_device.ancilla_offset != cmd_device.ancilla_offset
    assert cmd_device.ancilla_offset != file_device.ancilla_offset
    assert default_device.ancilla_offset != file_device.ancilla_offset
