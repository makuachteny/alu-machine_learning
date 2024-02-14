#!/usr/bin/env python3
''' This module contains a function for valid convolution on grayscale images '''


import numpy as np


def convolve_grayscale_valid(images, kernel):
    m, h, w = images.shape
    kh, kw = kernel.shape

    out_h = h - kh + 1
    out_w = w - kw + 1
    output = np.zeros((m, out_h, out_w))

    # Perform convolution on each pixel
    for i in range(out_h):
        for j in range(out_w):
            output[:, i, j] = np.sum(
                images[:, i:kh+i, j:kw+j] * kernel, axis=(1, 2)
            )
    return output
