#!/usr/bin/env python3
'''This module calculates the minor of a matrix'''


def cofactor(matrix):
    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square and non-empty
    num_rows = len(matrix)
    if num_rows == 0 or any(len(row) != num_rows for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Calculate cofactor matrix
    def minor(m, i, j):
        return [row[:j] + row[j + 1:] for row in (m[:i] + m[i + 1:])]

    def determinant(m):
        if len(m) == 1:
            return m[0][0]
        elif len(m) == 2:
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]
        else:
            det = 0
            for j in range(len(m)):
                det += ((-1) ** j) * m[0][j] * determinant(minor(m, 0, j))
            return det

    cofactor_matrix = []
    for i in range(num_rows):
        cofactor_row = []
        for j in range(num_rows):
            minor_matrix = minor(matrix, i, j)
            cofactor_row.append(((-1) ** (i + j)) * determinant(minor_matrix))
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix
