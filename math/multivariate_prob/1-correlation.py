#!/usr/bin/env python3
''' This module's function calculates a correlation of a matrix'''

import numpy as np


def correlation(C):
    '''Calculates the correlation matrix from a covariance matrix:
    Args: 
        C: A numpy.ndarray of shape(d, d) containing a covariance matrix.
    
    Returns: 
        A numpy.ndarray of shape(d,d) containing the correlation matrix.
    
    Raises: 
        TypeError: If C is not a numpy.ndarray.
        ValueError: If C is not a 2D square matrix
    '''
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        raise TypeError("C must be a 2D numpy.ndarray")

    if C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Calculate standard deviations
    std_dev = np.sqrt(np.diag(C))

    # Avoid division by zero (replace diagonals with 1s)
    std_dev = np.where(std_dev == 0, 1, std_dev)

    # Calculate correlation matrix (element-wise division)
    return C / np.outer(std_dev, std_dev)
