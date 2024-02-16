#!/usr/bin/env python 3
''' This module performs a convolution on grayscale images with custom padding '''


import numpy as np

def convolve_grayscale_padding(images, kernel, padding):
    ''' Function performs a convolution on grayscale images with custom padding '''

    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate the needed padding
    ph = max((kh - 1) // 2, kh // 2)
    pw = max((kw - 1) // 2, kw // 2)

    # calculate the padded images, the images should be padded with zeros

    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    
    ph_h = h
    pw_w = w
    # calculate the output height and width of the convolved image
    convolved_images = np.zeros(((m, ph_h, pw_w)))

    # Perform convolution on each pixel
    for i in range(ph_h):
        for j in range(pw_w):
            convolved_images[:, i, j] = np.sum(
                padded_images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
            )

    return convolved_images
