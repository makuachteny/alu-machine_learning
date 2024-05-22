#!/usr/bin/env python3
""" Forward Prop
"""
import tensorflow as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """ Function that builds the forward propagation graph for
        the neural network
    """
    # Initialize the input layer
    layer = x
    # Build the hidden layers
    for i in range(len(layer_sizes)):
        layer = create_layer(layer, layer_sizes[i], activations[i])

    return layer
