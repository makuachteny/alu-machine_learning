#!/usr/bin/env python3
'''This module performs convolution on images with multiple kernels'''

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    ''' This function performs convolution on images with multiple kernels '''
    m, h, w, c = images.shape
    kh, kw, c, nc = kernels.shape
    sh, sw = stride

    # Add padding according to the specified padding scheme/model
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded_images = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    ch = (h + 2 * ph - kh) // sh + 1
    cw = (w + 2 * pw - kw) // sw + 1

    convolved_images = np.zeros(
        (m, (h - kh + 2 * ph) // sh + 1, (w - kw + 2 * pw) // sw + 1, nc))

    for i in range(ch):
        for j in range(cw):
            for k in range(nc):
                convolved_images[:, i, j, k] = np.sum(
                    padded_images[:, i * sh:i * sh + kh, j *
                                  sw:j * sw + kw, :] * kernels[:, :, :, k],
                    axis=(1, 2, 3)
                )

    return convolved_images
