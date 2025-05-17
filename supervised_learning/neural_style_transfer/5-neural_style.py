#!/usr/bin/env python3
"""
Defines class NST that performs tasks for neural style transfer
"""

import numpy as np
import tensorflow as tf


class NST:
    # Define the layers used for style and content extraction
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initialize the NST class with style and content images, and weights
        for the style and content costs.
        """
        # Validate the style image
        if type(style_image) is not np.ndarray or len(style_image.shape) != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        # Validate the content image
        if type(content_image) is not np.ndarray or len(content_image.shape) != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        # Validate the dimensions of the style image
        style_h, style_w, style_c = style_image.shape
        content_h, content_w, content_c = content_image.shape
        if style_h <= 0 or style_w <= 0 or style_c != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        # Validate the dimensions of the content image
        if content_h <= 0 or content_w <= 0 or content_c != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        # Validate alpha and beta values
        if (type(alpha) is not float and type(alpha) is not int) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if (type(beta) is not float and type(beta) is not int) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        # Scale the style and content images
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

        # Load the VGG19 model and generate features
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels are in the range [0, 1]
        and its largest side is 512 pixels.
        """
        # Validate the input image
        if type(image) is not np.ndarray or len(image.shape) != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        h, w, c = image.shape
        if h <= 0 or w <= 0 or c != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        # Calculate new dimensions while maintaining aspect ratio
        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        # Resize and normalize the image
        resized = tf.image.resize(np.expand_dims(image, axis=0),
                                  size=(h_new, w_new),
                                  method=tf.image.ResizeMethod.BICUBIC)
        rescaled = tf.divide(resized, 255)
        rescaled = tf.clip_by_value(rescaled, 0, 1)
        return rescaled

    def load_model(self):
        """
        Loads the VGG19 model and extracts the outputs of the style and
        content layers.
        """
        # Load the pre-trained VGG19 model
        VGG19_model = tf.keras.applications.VGG19(include_top=False,
                                                  weights='imagenet')

        style_outputs = []
        content_output = None

        # Extract the outputs of the specified style and content layers
        for layer in VGG19_model.layers:
            if layer.name in self.style_layers:
                style_outputs.append(layer.output)
            if layer.name == self.content_layer:
                content_output = layer.output

            # Freeze the layers to prevent training
            layer.trainable = False

        # Combine the style and content outputs into a single model
        outputs = style_outputs + [content_output]
        self.model = tf.keras.models.Model(VGG19_model.input, outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """
        Computes the Gram matrix for a given input tensor.
        """
        # Validate the input tensor
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor of rank 4")
        if len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")
        _, h, w, c = input_layer.shape
        product = int(h * w)

        # Reshape the tensor and compute the Gram matrix
        features = tf.reshape(input_layer, (product, c))
        gram = tf.matmul(features, features, transpose_a=True)
        gram = tf.expand_dims(gram, axis=0)
        gram /= tf.cast(product, tf.float32)
        return gram

    def generate_features(self):
        """
        Preprocesses the style and content images and generates their
        respective features using the model.
        """
        VGG19_model = tf.keras.applications.vgg19
        preprocess_style = VGG19_model.preprocess_input(
            self.style_image * 255)
        preprocess_content = VGG19_model.preprocess_input(
            self.content_image * 255)

        # Extract style and content features
        style_features = self.model(preprocess_style)[:-1]
        content_feature = self.model(preprocess_content)[-1]

        # Compute the Gram matrices for the style features
        gram_style_features = []
        for feature in style_features:
            gram_style_features.append(self.gram_matrix(feature))

        self.gram_style_features = gram_style_features
        self.content_feature = content_feature

    def layer_style_cost(self, style_output, gram_target):
        """
        Computes the style cost for a single layer.
        """
        # Validate the style output tensor
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
           len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")
        one, h, w, c = style_output.shape

        # Validate the Gram target tensor
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
           len(gram_target.shape) != 3 or gram_target.shape != (1, c, c):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c, c))

        # Compute the style cost
        gram_style = self.gram_matrix(style_output)
        diff = tf.reduce_mean(tf.square(gram_style - gram_target))
        return diff

    def style_cost(self, style_outputs):
        """
        Computes the total style cost across all style layers.
        """
        length = len(self.style_layers)

        # Validate the style outputs
        if type(style_outputs) is not list or len(style_outputs) != length:
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    length))

        # Compute the weighted style cost
        weight = 1 / length
        style_cost = 0
        for i in range(length):
            style_cost += (
                self.layer_style_cost(style_outputs[i],
                                      self.gram_style_features[i]) * weight)
        return style_cost
