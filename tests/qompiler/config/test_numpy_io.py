import random
import tempfile

from quompiler.config.numpy_io import save_ndarray, load_ndarray
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_unitary

formatter = MatrixFormatter(precision=2)


def test_save_array(mocker):
    n = random.randint(2, 10)
    mat = random_unitary(n)

    mock_open = mocker.mock_open()
    mocker.patch("quompiler.config.config_manager.open", mock_open)
    mocker.patch("builtins.open", mock_open)
    file_path = "tmp_config.npy"

    # execute
    save_ndarray(file_path, mat, fmt="npy")

    # verify
    mock_open.assert_called_with(file_path, "wb")
    assert mock_open().write.call_count == 2


def test_load_array():
    n = random.randint(2, 10)
    mat = random_unitary(n)

    with tempfile.NamedTemporaryFile(suffix=".npy", mode="w+", delete=True) as tmp:
        save_ndarray(tmp.name, mat, fmt="npy")
        print(tmp.name)  # path to the temp file

        # execute
        ndarray = load_ndarray(tmp.name, fmt="npy")

        # verify
        assert ndarray.shape == (n, n)
        print(formatter.tostr(ndarray))
