#!/usr/bin/env python3

"""This module contains the function that
calculates normalization constants of a matrix"""

import numpy as np


def normalization_constants(X):
    """
    calculates the normalization (standardization)
    constants of a matrix
    X - is the numpy.ndarray of shape (m, nx) to normalize
        m - is the number of data points
        nx - is the number of features
    Returns: the mean and standard deviation of each feature
    """

    mean = X.mean(axis=0)
    std = X.std(axis=0)

    return mean, std
