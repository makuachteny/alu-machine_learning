#!/usr/bin/env python3
'''This function calculates the shape of a matrix'''
def matrix_shape(matrix):
    """ Calculates the shape of a matrix """
    shape = []
    while type(matrix) == list:
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape

mat1 = [[1, 2], [3, 4]]
print(matrix_shape(mat1))
mat2 = [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]]
print(matrix_shape(mat2))
