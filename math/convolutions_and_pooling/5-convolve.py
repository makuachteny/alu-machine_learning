#!/usr/bin/env python3
'''This module performs convolution on images with multiple kernels'''

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    ''' This function performs convolution on images with multiple kernels '''
    m, height, width, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride
    if padding is 'same':
        ph = ((((height - 1) * sh) + kh - height) // 2) + 1
        pw = ((((width - 1) * sw) + kw - width) // 2) + 1
    elif padding is 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    'constant', constant_values=0)
    ch = ((height + (2 * ph) - kh) // sh) + 1
    cw = ((width + (2 * pw) - kw) // sw) + 1
    convoluted = np.zeros((m, ch, cw, nc))

    for index in range(nc):
        kernel_index = kernels[:, :, :, index]
        for i, h in enumerate(range(0, (height + (2 * ph) - kh + 1), sh)):
            for j, w in enumerate(range(0, (width + (2 * pw) - kw + 1), sw)):
                output = np.sum(images[:, h: h + kh, w: w + kw, :]
                                * kernel_index, axis=(1, 2, 3))
                convoluted[:, i, j, index] = output

    return convoluted

