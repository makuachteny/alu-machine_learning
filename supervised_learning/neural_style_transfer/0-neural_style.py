#!/usr/bin/env python3
"""
Defines class NST that performs tasks for neural style transfer
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    Performs tasks for Neural Style Transfer
    """

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for Neural Style Transfer class
        """
        # Input validation
        if not isinstance(style_image, np.ndarray) or style_image.ndim != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or content_image.ndim != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if style_image.shape[2] != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "style_image and content_image must have 3 channels (RGB)")

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        # Ensure eager execution
        tf.config.run_functions_eagerly(True)

        # Set attributes
        self.style_image = NST.scale_image(style_image)
        self.content_image = NST.scale_image(content_image)
        self.alpha = float(alpha)
        self.beta = float(beta)

        # Load model
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixel values are between 0 and 1
        and its largest side is 512 pixels
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        if image.shape[2] != 3:
            raise TypeError("image must have 3 channels (RGB)")

        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        resized = tf.image.resize(np.expand_dims(image, axis=0),
                                  size=(h_new, w_new),
                                  method='bicubic')
        rescaled = resized / 255.0
        rescaled = tf.clip_by_value(rescaled, 0.0, 1.0)
        return tf.cast(rescaled, dtype=tf.float32)

    def load_model(self):
        """
        Creates the model used to calculate cost from VGG19 base model
        """
        base_model = tf.keras.applications.VGG19(include_top=False,
                                                 weights='imagenet')
        base_model.trainable = False

        style_outputs = []
        content_output = None

        for layer in base_model.layers:
            if layer.name in self.style_layers:
                style_outputs.append(layer.output)
            elif layer.name == self.content_layer:
                content_output = layer.output
            layer.trainable = False

        outputs = style_outputs + [content_output]
        self.model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
