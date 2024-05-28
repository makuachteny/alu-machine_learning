#!/usr/bin/ env python3
""" L2 Regularization layer"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ L2 Regularization layer"""

    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    l2 = tf.contrib.layers.l2_regularizer(lambtha)

    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=initializer,
                            kernel_regularizer=l2)

    return layer(prev)
