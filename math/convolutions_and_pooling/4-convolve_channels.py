#!/usr/bin/env python3
''' This module performs convolution on images with multiple channels '''

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    ''' Performs convolutions on images channels '''

    m, h, w, c = images.shape
    kh, kw, cc = kernel.shape
    sh, sw = stride

    # Calculate padding
    if padding == 'same':
        ph = max((h - 1) * sh + kh - h, 0) // 2
        pw = max((w - 1) * sw + kw - w, 0) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Pad images
    padded_images = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    # Calculate output dimensions
    if padding == 'valid':
        ch = (h - kh) // sh + 1
        cw = (w - kw) // sw + 1
    else:
        ch = (h + 2 * ph - kh) // sh + 1
        cw = (w + 2 * pw - kw) // sw + 1

    # Initialize convolved images
    convolved_images = np.zeros((m, ch, cw))

    # Perform convolution
    for i in range(ch):
        for j in range(cw):
            convolved_images[:, i, j] = np.sum(
                padded_images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :] * kernel,
                axis=(1, 2, 3)
            )

    return convolved_images
