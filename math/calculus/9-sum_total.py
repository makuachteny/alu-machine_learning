#!/usr/bin/env python3
''' This module calculates the total sum of a summation'''


def summation_i_squared(n):
    '''This function calculates the total sum of a summation'''
    # Check if n is valid number
    if not isinstance(n, int) or n < 1:
        return None
    elif n == 1:
        return 1
    else:
        return n**2 + summation_i_squared(n - 1)


# Example usage:
result = summation_i_squared(5)
print(result)
