#!/usr/bin/env python3
'''This functions flips the matrix over its diagonals/ Transposes the matrix'''


def matrix_transpose(matrix):
    '''This function transposes the matrix'''
    transposed = []
    for i in range(len(matrix[0])):
        transposed.append([])
        for j in range(len(matrix)):
            transposed[i].append(matrix[j][i])
    return transposed
