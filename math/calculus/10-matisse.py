#!/usr/bin/env python3
'''
    This module calculates the derivative of a polynomial
'''


def poly_derivative(poly):
    '''
        calculates the derivative of  polynomials
    '''
    if not isinstance(poly, list) or len(poly) <= 1:
        return [0]
    return [poly[i] * i for i in range(1, len(poly))]


poly = [5, 3, 0, 1]
print(poly_derivative(poly))
