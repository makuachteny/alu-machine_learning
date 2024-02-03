#!/usr/bin/env python3
'''Calculates the integral of a polynomial.'''


def poly_integral(poly, C=0):
    """
    function that integrate a polynomialcoefficients
    """
    if not isinstance(poly, list) or len(poly) < 1 or not (isinstance(C, int) or isinstance(C, float)):
        return None
    if isinstance(C, float) and C.is_integer():
        C = int(C)
    integral = [C]
    for power, coefficient in enumerate(poly):
        new_coefficient = coefficient // (power + 1) if coefficient % (
            power + 1) == 0 else coefficient / (power + 1)
        integral.append(new_coefficient)
    while integral and integral[-1] == 0:
        integral = integral[:-1]
    return integral
