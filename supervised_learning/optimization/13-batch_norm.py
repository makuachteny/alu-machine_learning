#!/usr/bin/env python3

"""This module contains the function that
normalizes an unactivated output of a neural network
using batch normalization:
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    normalizes an unactivated output of a neural network
    using batch normalization
    Z - is a numpy.ndarray of shape (m, n) that should be normalized
        m - is the number of data points
        n - is the number of features in Z
    gamma - is a numpy.ndarray of shape (1, n) with the scales used for
    batch normalization
    beta - is a numpy.ndarray of shape (1, n) with the offsets used for
    batch normalization
    epsilon - is a small number used to avoid division by zero
    Returns: The normalized Z matrix"""

    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Znorm = (Z - mean) / ((variance + epsilon) ** 0.5)
    Znorm = (gamma * Znorm) + beta
    return Znorm
