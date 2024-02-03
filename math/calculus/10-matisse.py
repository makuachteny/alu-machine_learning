#!/usr/bin/env python3
'''
    This module calculates the derivative of a polynomial
'''


def poly_derivative(poly):
    '''
        calculates the derivative of  polynomials
    '''
    if not isinstance(poly, list) or len(poly) <= 1:
        return [None]
    return [poly[i] * i for i in range(1, len(poly))]
