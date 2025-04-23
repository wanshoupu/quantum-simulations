from sympy import Matrix, pretty, Symbol

default_columns = 100


def mat_format(mat: Matrix):
    print()
    blank = Symbol(' ')  # blank is valid symbol name
    # Replace zeros with '' for display
    s = mat.applyfunc(lambda x: blank if x == 0 else x)
    num_columns = default_columns
    while pretty(s[0, :], num_columns=num_columns).count('↪'):
        num_columns <<= 1
    return pretty(s, num_columns=num_columns)


def mat_print(mat: Matrix):
    print(mat_format(mat))
