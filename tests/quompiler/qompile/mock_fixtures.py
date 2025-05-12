from quompiler.qompile.configure import QompilerConfig


def mock_device(mocker, aspace):
    device_cls = mocker.patch("quompiler.qompile.configure.DeviceConfig")
    result = device_cls.return_value
    result.arange = [aspace, 200]
    return result


def mock_config(mocker, emit: str = "CLIFFORD_T", aspace=1) -> QompilerConfig:
    config_cls = mocker.patch("quompiler.qompile.configure.QompilerConfig")

    result = config_cls.return_value
    result.device = mock_device(mocker, aspace)
    result.target = "CIRQ"
    result.emit = emit

    return result


def mock_decompose(mocker):
    mock_std_decompose = mocker.patch("quompiler.utils.std_decompose.std_decompose", return_value="mocked data")
    mock_ctrl_decompose = mocker.patch("quompiler.utils.std_decompose.ctrl_decompose", return_value="mocked data")
    mock_std_decompose.assert_called_once()
    mock_ctrl_decompose.assert_called_once()
