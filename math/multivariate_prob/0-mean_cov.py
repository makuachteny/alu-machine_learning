import numpy as np

#!/usr/bin/env python3
''' This module contains a function that calculates the mean and covariance of a dataset'''

import numpy as np


def mean_cov(X):
    ''' Calculates mean and covariance of a data set'''
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0)
    centered_X = X - mean
    cov = np.dot(centered_X.T, centered_X) / (n - 1)

    return mean.reshape(1, d), cov
