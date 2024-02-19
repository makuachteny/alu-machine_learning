#!/usr/bin/env python3
''' This module performs pooling on images '''


import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    ''' This function performs pooling on images '''
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1
    out = np.zeros((m, out_h, out_w, c))
    for i in range(out_h):
        for j in range(out_w):
            if mode == 'max':
                out[:, i, j, :] = np.max(
                    images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                    axis=(1, 2))
            elif mode == 'avg':
                
                out[:, i, j, :] = np.mean(
                    images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                    axis=(1, 2))
    return out