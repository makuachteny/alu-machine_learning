#!/usr/bin/env python3
"""
Module that performs the Expectation Maximization for a GMM
"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Function that performs the Expectation Maximization for a GMM
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) != int or k <= 0 or X.shape[0] < k:
        return None, None, None, None, None
    if type(iterations) != int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) != float or tol < 0:
        return None, None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    loglikelihood = 0
    i = 0
    while i < iterations:
        g, loglikelihood_new = expectation(X, pi, m, S)
        if verbose is True and (i % 10 == 0):
            print("Log Likelihood after {} iterations: {}".format(
                i, loglikelihood_new.round(5)))
        if abs(loglikelihood_new - loglikelihood) <= tol:
            break
        pi, m, S = maximization(X, g)
        i += 1
        loglikelihood = loglikelihood_new
    g, loglikelihood_new = expectation(X, pi, m, S)
    if verbose is True:
        print("Log Likelihood after {} iterations: {}".format(
            i, loglikelihood_new.round(5)))
    return pi, m, S, g, loglikelihood_new
