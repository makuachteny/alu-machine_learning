#!/usr/bin/env python3
'''
This module contains a function that calculates the determinant of a matrix.
'''


def determinant(matrix):
    '''This function calculates the determinant of a matrix.'''

    # Check if the matrix is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError('matrix must be a list of lists')

    # Check if the matrix is a 0x0 matrix
    if len(matrix) == 0:
        return 1

    # Check if matrix is square
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    # Calculate the determinant of the matrix
    if len(matrix) == 1:
        return matrix[0][0]
    elif len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        det = 0
        for i in range(len(matrix)):
            minor = [row[:i] + row[i+1:] for row in matrix[1:]]
            det += matrix[0][i] * ((-1) ** i) * determinant(minor)
        return det
