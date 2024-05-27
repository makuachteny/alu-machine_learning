#!/usr/bin/env python3
""" Module contains a function that calculates the cost of a neural network with L2 regularization """


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ function calculates the cost of a neural network with L2.
    Parameters: 
    cost: the cost of the network without L2 regularization.
    lambtha: the regularization parameter.
    weights: a dictionary of the weights and biases of the neural network.
    L: the number of layers in the network.
    m: the number of data points used.
    Returns: the cost of the network accounting for L2 regularization.
    """
    # Initialize the L2 regularization term to zero
    l2_reg = 0
    # Loop through the layers to accumulate the L2 regularization term    
    for i in range(1, L+1):
        # Get the weight matrix for layer i
        w = weights['W' + str(i)]

        # Compute the sum of the squares of the weights for layer i
        reg_weights = np.sum(w ** 2)

        # Accumulate the regulatization term, scaled by lambda / (2 * m)       
        l2_reg += reg_weights * (lambtha / (2 * m))

    cost += l2_reg
    
    return cost
