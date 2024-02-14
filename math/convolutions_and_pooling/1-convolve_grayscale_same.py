#!/usr/bin/env python3
''' This module performs same convolution on grayscale images '''

import numpy as np


def convolve_grayscale_same(images, kernel):
    ''' This function performs same convolution on grayscale images'''

    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate the amount of padding needed
    ph = max((kh - 1) // 2, kh // 2)
    pw = max((kw - 1) // 2, kw // 2)

    # Calculate the padded images
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    padded_h = h
    padded_w = w
    convolved_images = np.zeros((m, padded_h, padded_w))

    # Perform convolution on each pixel
    for i in range(padded_h):
        for j in range(padded_w):
            convolved_images[:, i, j] = np.sum(
                padded_images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
            )
    return convolved_images
