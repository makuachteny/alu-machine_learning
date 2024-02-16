#!/usr/bin/env python3
''' This module performsn convolution on grayscale images'''

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    ''' Function covolves images '''

    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Add padding according to the specified padding scheme/mode
    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding[0], padding[1]

    # Calculate the padded images
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    # Calculate the height and width of the convolved images
    padded_h = h + 2 * ph
    padded_w = w + 2 * pw

    ch = int((padded_h - kh) / sh) + 1
    cw = int((padded_w - kw) / sw) + 1

    # calculate the convolved images
    convolved_images = np.zeros((m, ch, cw))

    # Perform convolution on each pixel
    for i in range(ch):
        for j in range(cw):
            roi = padded_images[:, i*sh:i*sh+kh, j*sw:j*sw+kw]
            convolved_images[:, i, j] = np.sum(roi * kernel, axis=(1, 2))

    return convolved_images
