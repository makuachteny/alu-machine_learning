#!/usr/bin/env python3
"""Module defines a function that creates a layer"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a dense layer using the previous layer with He et al. initialization.
    Parameters: 
    - prev: The output tensor from the previous layer
    - n: The number of nodes in the layer to create
    - activation: The activation function that the layer should use
    Returns: The output tensor of the layer
    """

    # Initialize weights using He et al. initializer
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    # Create the dense layer
    layer = tf.layers.dense(input=prev, units=n, activation=activation, kernel_initializer=init, name='layer')

    return layer
