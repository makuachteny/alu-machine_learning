#!/usr/bin/env python3
'''This function performs element-wise addition, subtraction, multiplication, and division of two matrices'''


def np_elementwise(mat1, mat2):
    '''Function performs element-wise addition, subtraction, multiplication, and division of two matrices'''
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
