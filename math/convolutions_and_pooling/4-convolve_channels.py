#!/usr/bin/env python3
''' This module performs convolution on images with multiple channels '''

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    ''' Performs convolutions on images channels '''

    m, height, width, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((((height - 1) * sh) + kh - height) // 2) + 1
        pw = ((((width - 1) * sw) + kw - width) // 2) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding

    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    'constant', constant_values=0)

    ch = ((height + (2 * ph) - kh) // sh) + 1
    cw = ((width + (2 * pw) - kw) // sw) + 1
    convoluted = np.zeros((m, ch, cw))

    for i in range(ch):
        for j in range(cw):
            h = i * sh
            w = j * sw
            output = np.sum(images[:, h: h + kh, w: w + kw, :] * kernel,
                            axis=(1, 2, 3))
            convoluted[:, i, j] = output

    return convoluted
