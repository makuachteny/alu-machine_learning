#!/usr/bin/env python3
''' Convolution on grayscale images with custom padding '''


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    ''' Function performs covolves images with custom padding '''

    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate the needed padding
    ph = max((kh - 1) // 2, kh // 2)
    pw = max((kw - 1) // 2, kw // 2)

    # calculate the padded images, the images should be padded with zeros
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    # calculate the padded height and width of the convolved images
    padded_h = h + 2 * ph
    padded_w = w + 2 * pw

    # calculate the output height and width of the convolved image
    convolved_images = np.zeros(((m, padded_h, padded_w)))

# Perform convolution on each pixel
    for i in range(padded_h - kh + 1):
        for j in range(padded_w - kw + 1):
            # Extract the region of interest (ROI) from the padded images
            roi = padded_images[:, i:i+kh, j:j+kw]
            # Apply convolution by element-wise multiplication and summation
            convolved_images[:, i, j] = np.sum(roi * kernel, axis=(1, 2))

    return convolved_images
