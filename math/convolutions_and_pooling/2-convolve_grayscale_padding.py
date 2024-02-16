#!/usr/bin/env python3
''' Convolution on grayscale images with custom padding '''

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    ''' Function performs covolves images with custom padding '''

    m, h, w = images.shape
    kh, kw = kernel.shape

    # Add some padding based on the specified padding mode
    if padding == 'same':
        ph = max((kh - 1), 0)
        pw = max((kw - 1), 0)
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding

    # calculate the padded images, the images should be padded with zeros
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    # calculate the output height and width of the convolved images
    padded_h = h + 2 * ph
    padded_w = w + 2 * pw
    ch = padded_h - kh + 1
    cw = padded_w - kw + 1

    # calculate the output height and width of the convolved image
    convolved_images = np.zeros((m, ch, cw))

    # Perform convolution on each pixel
    for i in range(ch):
        for j in range(cw):
            # Extract the region of interest (ROI) from the padded images
            roi = padded_images[:, i:i+kh, j:j+kw]
            # Apply convolution by element-wise multiplication and summation
            convolved_images[:, i, j] = np.sum(roi * kernel, axis=(1, 2))

    return convolved_images
