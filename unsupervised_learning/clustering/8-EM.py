#!/usr/bin/env python3
""" Module performs the Expectation-Maximization (EM) algorithm for a GMM """

import numpy as np
# Import the necessary functions from their respective modules
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization

def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the Expectation-Maximization (EM) algorithm for a GMM
    """
    n, d = X.shape
    
    # Initialize parameters
    pi, m, S = initialize(X, k)
    
    l = 0  # Initial log likelihood
    for i in range(iterations):
        # E-step
        g, l_new = expectation(X, pi, m, S)
        
        # M-step
        pi, m, S = maximization(X, g)
        
        # Check for convergence
        if abs(l_new - l) <= tol:
            break
        
        l = l_new
        
        # Verbose logging
        if verbose and i % 10 == 0:
            print(f"Log Likelihood after {i} iterations: {l:.5f}")
    
    # Final log likelihood print
    if verbose:
        print(f"Log Likelihood after {i} iterations: {l:.5f}")

    return (pi, m, S, g, l) if l > -np.inf else (None, None, None, None, None)
