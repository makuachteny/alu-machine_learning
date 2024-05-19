#!/usr/bin/env python3
""" Module defines a function that returns two placeholders, x and y, for neural network:
    nx: the number of input features to the neuron
    classes: the number of classes in our classifier
"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """ Function that returns two placeholders, x and y, for neural network:
        nx: the number of input features to the neuron
        classes: the number of classes in our classifier
    """
    x = tf.keras.Input(shape=(nx,), name="x")
    y = tf.keras.Input(shape=(classes,), name="y")
    return x, y
