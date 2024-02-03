#!/usr/bin/env python3
'''
    This module defines a function that calculates the summation
    of numbers from 1 to n
'''


def summation_i_squared(n):
    '''
    calculates the summation
    squares of numbers from 1 to n
    '''
    if type(n) is not int or n < 1:
        return None
    summation = (n * (n + 1) * ((2 * n) + 1)) / 6
    return int(summation)
