#!/usr/bin/env python3
""" L2 Regularization Gradient Descent
"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent with L2 regularization.

    Parameters:
    Y (numpy.ndarray): One-hot numpy.ndarray of shape (classes, m) that contains the correct labels for the data.
    weights (dict): A dictionary of the weights and biases of the neural network.
    cache (dict): A dictionary of the outputs of each layer of the neural network.
    alpha (float): The learning rate.
    lambtha (float): The L2 regularization parameter.
    L (int): The number of layers of the network.

    Returns:
    None: The weights and biases are updated in place.
    """
    m = Y.shape[1]  # Number of data points

    # Calculate the gradient of the cost with respect to the output of the last layer
    A_L = cache['A' + str(L)]
    dZ_L = A_L - Y  # Derivative of the loss with respect to A_L
    dW_L = (1 / m) * np.dot(dZ_L,
                            cache['A' + str(L - 1)].T) + (lambtha / m) * weights['W' + str(L)]
    db_L = (1 / m) * np.sum(dZ_L, axis=1, keepdims=True)

    # Update the weights and biases of the last layer
    weights['W' + str(L)] -= alpha * dW_L
    weights['b' + str(L)] -= alpha * db_L

    # Backpropagate through the layers
    dZ_curr = dZ_L
    for l in range(L - 1, 0, -1):
        A_curr = cache['A' + str(l)]
        A_prev = cache['A' + str(l - 1)]

        # Calculate gradients
        # Derivative of the tanh activation
        dZ_curr = np.dot(weights['W' + str(l + 1)].T,
                         dZ_curr) * (1 - A_curr ** 2)
        dW_curr = (1 / m) * np.dot(dZ_curr, A_prev.T) + \
            (lambtha / m) * weights['W' + str(l)]
        db_curr = (1 / m) * np.sum(dZ_curr, axis=1, keepdims=True)

        # Update weights and biases
        weights['W' + str(l)] -= alpha * dW_curr
        weights['b' + str(l)] -= alpha * db_curr

    return None