#!/usr/bin/env python3
"""Module defines a function that creates a layer"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """Creates a dense layer using previous layer."""
    # Initialize weights using He et al. initializer
    W_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    # Create the dense layer with activation function
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=W_init, name='layer')

    return layer(prev)
