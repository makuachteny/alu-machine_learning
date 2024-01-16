#!/usr/bin/env python3
'''This function concatenates two matrices along a specific axis'''


def cat_matrices(mat1, mat2, axis=0):
    '''This function concatenates two matrices along a specific axis'''
    if axis == 0:
        return mat1 + mat2
    elif axis == 1:
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    else:
        return None
