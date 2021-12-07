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
    mat = apply_permutations(mat)

    # eliminate below the diagonal
    for j in range(n):
        # if free variable, can't eliminate column
        if j > m - 1:
            continue
        if mat[j][j] == 0:
            swap_row_index = nonzero_below_row_index(mat, j, j)
            if swap_row_index != -1:
                mat = permute(mat, j, swap_row_index)
            else:
                continue

        for i in range(j+1, m):
            # if value is non-zero, eliminate using row operations
            if v != 0:
                # get a constant multiple of the pivot row for current column, update current row
                factor = mat[i][j] / mat[j][j]
                pivot_row_multiple = [pv * factor for pv in mat[j]]
                mat[i] = [rv - prmv for rv, prmv in zip(mat[i], pivot_row_multiple)]

    # eliminate above the diagonal
    for j in range(n):
        # if free variable, can't eliminate column
        if j > m - 1:
            continue
        if mat[j][j] == 0:
            swap_row_index = nonzero_below_row_index(mat, j, j)
            if swap_row_index != -1:
                mat = permute(mat, j, swap_row_index)
            else:
                continue

        for i in range(j):
            # if value is non-zero, eliminate using row operations
            if v != 0:
                # get a constant multiple of the pivot row for current column, update current row
                factor = mat[i][j] / mat[j][j]
                pivot_row_multiple = [pv * factor for pv in mat[j]]
                mat[i] = [rv - prmv for rv, prmv in zip(mat[i], pivot_row_multiple)]

    try:
        # extract solution vector from ref augmented matrix
        x = []
        for i in range(m):
            if all([v == 0 for v in mat[i][:-1]]):
                # if there is a zero row with non-zero augmented value, there is no solution
                if mat[i][-1] != 0:
                    return []
                # if there is a zero row with zero augmented value, that variable is free (infinite solutions)
                x.append(f'x_{i+1}')
                continue

            coeff = mat[i][i]
            coeff_col = i
            while coeff == 0:
                x.append(f'x_{coeff_col+1}')
                coeff_col += 1
                if coeff_col == len(mat[i]):
                    continue
                coeff = mat[i][coeff_col]

            curr_x = mat[i][n] / coeff
            for j, v in enumerate(mat[i][:-1]):
                if v != 0 and j != coeff_col:
                    val_str = f' + {-v/coeff} x_{j+1}'
                    val_str = val_str.replace('+ -', '- ')
                    curr_x = str(curr_x) + val_str
            x.append(curr_x)

        while len(x) < n:
            x.append(f'x_{len(x)+1}')
        return x
    except ZeroDivisionError:
        raise ValueError('Invalid augmented matrix (divide by zero).')


# returns the index of the first row (ind) below row (i) in column (j) with non-zero mat_{ind,j}
def nonzero_below_row_index(mat, i, j):
    for ind in range(i+1, len(mat)):
        if mat[ind][j] != 0:
            return ind
    return -1


# applies permutation transformations to a matrix to get non-zero values on the diagonal
def apply_permutations(mat):
    for i in range(len(mat)):
        if mat[i][i] == 0:
            for j in range(i+1, len(mat)):
                if mat[j][i] != 0:
                    return apply_permutations(permute(mat, i, j))
    return mat


# performs row permutation P(i,j) to mat
def permute(mat, i, j):
    return np.matmul(permutation(len(mat), i, j), mat)


# defines row permutation P(i,j)_n
def permutation(n, i, j):
    P = identity(n)
    temp = P[i]
    P[i] = P[j]
    P[j] = temp
    return P


# defines identity matrix I_n
def identity(n):
    I = []
    for i in range(n):
        I.append([1 if i == j else 0 for j in range(n)])
    return I


# augments mat with v
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
            print('-----')
            if len(x) > 0:
                for i, xi in enumerate(x):
                    print(f'x_{i+1} = {xi}')
            else:
                print('No solution.')
            print('-----\n')
        except ValueError as e:
            print(f'{e}\n')
            continue


if __name__ == '__main__':
    main()
