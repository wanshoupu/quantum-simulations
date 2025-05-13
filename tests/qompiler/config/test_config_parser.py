import sys

import pytest

from quompiler.config.config_parser import ConfigManager, merge_dicts


def test_parse_default_config():
    parser = ConfigManager()
    config = parser.get_config()
    assert config.device is not None
    assert config.target == "CIRQ"
    assert config.emit == "SINGLET"


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
    parser = ConfigManager()

    # execute
    parser.parse_args()
    actual = parser.get_config()

    # verify
    assert actual.target == target
    assert actual.rtol == rtol
    assert actual.atol == atol
    assert actual.device.ancilla_offset == offset


def test_save(mocker):
    test_args = ['--emit', 'UNIV_GATE', "--rtol", "1e-6", "--atol", "1e-9", "--ancilla_offset", "7"]
    mocker.patch('sys.argv', test_args)
    parser = ConfigManager()
    parser.parse_args()
    assert parser.config

    mock_open = mocker.mock_open()
    mocker.patch("quompiler.config.config_parser.open", mock_open)
    mocker.patch("builtins.open", mock_open)
    file_path = "tmp_config.json"

    # execute
    parser.save(file_path)

    # verify
    assert mock_open().write.call_count > 0
