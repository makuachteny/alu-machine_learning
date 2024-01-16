#!/usr/bin/env python3
'''This function concatenates two arrays, assuming that the arrays are lists of ints/floats'''


def cat_arrays(arr1, arr2):
    '''This function concatenates two arrays'''
    if len(arr1) == len(arr2):
        return [arr1[i] + arr2[i] for i in range(len(arr1))]
    else:
        return None
