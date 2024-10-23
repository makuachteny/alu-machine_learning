#!/usr/bin/env python3
"""
PCA
"""
import numpy as np

def pca(X, var=0.95):
    """
    Performs PCA on a dataset.
    
    Parameters:
    X (numpy.ndarray): The dataset of shape (n, d)
    var (float): The fraction of the variance that the PCA transformation should maintain
    
    Returns:
    numpy.ndarray: The weights matrix, W, that maintains var fraction of X's original variance
    """
    u, s, v = np.linalg.svd(X)
    ratios = list(x / np.sum(s) for x in s)
    variance = np.cumsum(ratios)
    nd = np.argwhere(variance >= var)[0, 0]
    W = v.T[:, :(nd + 1)]
    return (W)
    
    return W
