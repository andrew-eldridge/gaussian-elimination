# ge.py
# -----
# Gaussian Elimination in Python
# MATH 544 Final Project - Andrew Eldridge, Laura Witzel

import numpy as np


def ge(mat, v):
    # validate dimensions
    m = len(mat)
    n = len(mat[0])
    if len(v) != m:
        raise ValueError(
            f'Invalid dimensions for matrix-vector pairing: {len(v)}-vector cannot augment {m}x{n} matrix.')

    # augment matrix
    mat = augment(mat, v)

    # apply row permutations
    print(mat)
    mat = permute(mat)
    print(mat)

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
        print(mat)
        # extract solution vector from ref augmented matrix
        x = []
        for i in range(len(mat)):
            if all([v == 0 for v in mat[i][:-1]]):
                # if there is a zero row with non-zero augmented value, there is no solution
                if mat[i][-1] != 0:
                    return []
                # if there is a zero row with zero augmented value, that variable is free (infinite solutions)
                x.append(f'x_{i+1}')
                continue

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


def permute(mat):
    for i in range(len(mat)):
        if mat[i][i] == 0:
            for j in range(i+1, len(mat)):
                if mat[i][j] != 0:
                    return permute(np.matmul(permutation(len(mat), i, j), mat))
    return mat


def permutation(n, i, j):
    P = identity(n)
    temp = P[i]
    P[i] = P[j]
    P[j] = temp
    return P


def identity(n):
    I = []
    for i in range(n):
        I.append([1 if i == j else 0 for j in range(n)])
    return I


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
