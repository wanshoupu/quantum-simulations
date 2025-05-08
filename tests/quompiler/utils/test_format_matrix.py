import numpy as np

from quompiler.construct.types import UnivGate
from quompiler.utils.format_matrix import MatrixFormatter


def test_intify_int():
    x = 1 + 1e-8
    formatter = MatrixFormatter()
    d = formatter.intify(x)
    assert isinstance(d, int)
    assert d == 1


def test_intify_negative_int():
    x = -1 + 1e-9
    formatter = MatrixFormatter()
    d = formatter.intify(x)
    assert isinstance(d, int)
    assert d == -1


def test_intify_nonint():
    x = -1 + 1e-4
    formatter = MatrixFormatter()
    d = formatter.intify(x)
    assert isinstance(d, float)
    assert d == x


def test_nformat_zero():
    c = 1e-8 + 1e-8 * 1j
    formatter = MatrixFormatter()
    d = formatter.nformat(c)
    assert isinstance(d, int)
    assert d == 0


def test_nformat_positive_real():
    c = 1.2 + 1e-8 * 1j
    formatter = MatrixFormatter()
    d = formatter.nformat(c)
    assert isinstance(d, float)
    assert d > 0


def test_nformat_negative_real():
    c = -1.2 + 1e-8 * 1j
    formatter = MatrixFormatter()
    d = formatter.nformat(c)
    assert isinstance(d, float)
    assert d < 0


def test_nformat_positive_imag():
    c = 1e-8 + .8j
    formatter = MatrixFormatter()
    d = formatter.nformat(c)
    assert isinstance(d, complex)
    assert d.real == 0
    assert d.imag > 0
    assert repr(d) == '0.8j'


def test_nformat_negative_imag():
    c = 1e-8 - .8j
    formatter = MatrixFormatter()
    d = formatter.nformat(c)
    assert isinstance(d, complex)
    assert d.real == 0
    assert d.imag < 0
    assert repr(d) == '(-0-0.8j)'


def test_1darray():
    array = np.array([1.0, 2.345678, 3 + 0j, 4 + 5j, -3 - 0j, -4 - 5j, -2.0, -5.5, +2.0, -5.5], dtype=np.complexfloating)
    s = formatter.tostr(array)
    # print(s)
    assert s == '[      1    2.35       3  (4+5j)      -3 (-4-5j)      -2    -5.5       2    -5.5]'


def test_gate_mat():
    m = -1j * UnivGate.H.matrix
    d = formatter.nformat(m[0, 0])
    print()
    print(d)
    assert isinstance(d, complex)
    assert d.real == 0
    assert d.imag < 0


def test_precision():
    matrix = np.array([[1.0, 2.345678], [3 + 0j, 4 + 5j], [-3 - 0j, -4 - 5j], [-2.0, -5.5], [+2.0, -5.5]], dtype=np.complexfloating)
    formatter = MatrixFormatter(precision=2)
    print(formatter.mformat(matrix))
    print(formatter.tostr(matrix, indent=10))


def test_tostr():
    formatter = MatrixFormatter()
    # Example shape, change as needed
    # Generate random binary matrix (0s and 1s)
    binary_matrix = np.random.randint(0, 2, (4, 4))

    # Convert to complex type
    complex_binary_matrix = binary_matrix.astype(np.complexfloating)

    # Print the result
    print(complex_binary_matrix)
    print(formatter.mformat(complex_binary_matrix))
    print(formatter.tostr(complex_binary_matrix, indent=10))
