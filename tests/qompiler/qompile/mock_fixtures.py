from quompiler.config.construct import QompilerConfig


def mock_device(mocker, ancilla_offset):
    device_cls = mocker.patch("quompiler.config.construct.DeviceConfig")
    result = device_cls.return_value
    result.ancilla_offset = ancilla_offset
    return result


def mock_config(mocker, emit: str = "SINGLET", ancilla_offset=1) -> QompilerConfig:
    config_cls = mocker.patch("quompiler.config.construct.QompilerConfig")

    result = config_cls.return_value
    result.device = mock_device(mocker, ancilla_offset)
    result.target = "CIRQ"
    result.emit = emit

    return result


def mock_decompose(mocker):
    mock_std_decompose = mocker.patch("quompiler.utils.std_decompose.std_decompose", return_value="mocked data")
    mock_ctrl_decompose = mocker.patch("quompiler.utils.std_decompose.ctrl_decompose", return_value="mocked data")
    mock_std_decompose.assert_called_once()
    mock_ctrl_decompose.assert_called_once()
