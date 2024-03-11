#!/usr/bin env python3

"""This module calculates the likelihood of data given a model"""

import numpy as np


def likelihood(x, n, p):
    """Calculates the likelihood of obtaining the data
    x: number of patients that develop severe side effects
    n: total number of patients observed
    p: is a 1D numpy.ndarray containing the various hypothetical probabilities of developing severe side effects
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("X must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("X cannot be greater than n")
    if not isinstance(P, np.ndarray) 