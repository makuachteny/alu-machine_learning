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

    # Loop through each layer of the neural network and update the weights and biases
    for i in range(L, 0, -1):
        # Retrieve the activations and parameters for the current layer
        A = cache["A" + str(i)]
        A_prev = cache["A" + str(i - 1)]
        W = weights["W" + str(i)]
        b = weights["b" + str(i)]

        # Compute the error term for the output layer
        if i == L:
            dZ = A - Y
        else:
            # Compute the error term for the hidden layers
            dZ = dA * (A * (1 - A))

        # Compute the gradients for the weights and biases
        db = dZ.mean(axis=1, keepdims=True)
        dW = np.matmul(dZ, A_prev.T) / m
        dA = np.matmul(W.T, dZ)

        # Update the weights and biases
        weights["W" + str(i)] -= alpha * dW
        weights["b" + str(i)] -= alpha * db

    return weights
