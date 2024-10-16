#!/usr/bin/env python3
'''This module provides a function for concatinating matrices'''


import numpy as np


def np_cat(mat1, mat2, axis=0):
    '''This function concatenates two matrices along a specific axis'''
    return np.concatenate((mat1, mat2), axis)

