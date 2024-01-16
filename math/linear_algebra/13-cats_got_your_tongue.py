#!/usr/bin/env python3
import numpy as np
'''This function concatenates two matrices along a specific axis using Numpy library'''


def np_cat(mat1, mat2, axis=0):
    '''This function concatenates two matrices along a specific axis using Numpy library'''
    return np.concatenate((mat1, mat2), axis)
