import numpy as np

from quompiler.utils.format_matrix import MatrixFormatter


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
