# ge.py
# -----
# Gaussian Elimination in Python
# MATH 544 Final Project - Andrew Eldridge, Laura Witzel


def ge(mat, v):
    # validate dimensions
    m = len(mat)
    n = len(mat[0])
    if len(v) != m:
        raise ValueError(
            f'Invalid dimensions for matrix-vector pairing: {len(v)}-vector cannot augment {m}x{n} matrix.')

    # augment matrix
    mat = augment(mat, v)

    # iteratively perform gaussian elimination
    for j in range(n):
        for i in range(m):
            # diagonal values and free variables don't need to be eliminated
            if i == j or j > m-1 or mat[j][j] == 0:
                continue
            # if a non-diagonal value is non-zero, eliminate using row operations
            if v != 0:
                # get a constant multiple of the pivot row for current column, update current row
                factor = mat[i][j] / mat[j][j]
                pivot_row_multiple = [pv * factor for pv in mat[j]]
                mat[i] = [rv - prmv for rv, prmv in zip(mat[i], pivot_row_multiple)]

    try:
        # extract solution vector from ref augmented matrix
        x = []
        for i in range(len(mat)):
            # if there is a zero row with non-zero augmented value, there is no solution
            if all([v == 0 for v in mat[i][:-1]]) and mat[i][-1] != 0:
                return []

            coeff = mat[i][i]
            curr_x = mat[i][n] / coeff
            for j, v in enumerate(mat[i][:-1]):
                if v != 0 and j != i:
                    val_str = f' + {-v/coeff} x_{j+1}'
                    val_str = val_str.replace('+ -', '- ')
                    curr_x = str(curr_x) + val_str
            x.append(curr_x)
        return x
    except ZeroDivisionError:
        raise ValueError('Invalid augmented matrix (divide by zero).')


def augment(mat, v):
    for i in range(len(mat)):
        mat[i].append(v[i])
    return mat


def main():
    print('-----\nGaussian elimination in Python - Andrew Eldridge, Laura Witzel\nType "exit" at any point to quit.\n-----')
    while True:
        # get matrix
        usr_in = input('Enter a matrix with semicolon-delimited rows and space-delimited columns (a11 a12; a21 a22): ')
        if usr_in == 'exit':
            print('Goodbye!')
            exit(0)

        try:
            # parse input into matrix
            rows = usr_in.split('; ')
            mat = []
            for row in rows:
                mat.append([float(v) for v in row.split(' ')])
            print(mat)

            # get vector
            usr_in = input('Enter a space-delimited vector (v1 v2 v3): ')
            if usr_in == 'exit':
                print('Goodbye!')
                exit(0)

            # parse input into vector
            v = [float(i) for i in usr_in.split(' ')]
            print(v)

            # perform gaussian elimination
            x = ge(mat, v)
            if len(x) > 0:
                print(f'-----\nx = {x}\n-----\n')
            else:
                print('No solution.')
        except ValueError as e:
            print(f'{e}\n')
            continue


if __name__ == '__main__':
    main()
