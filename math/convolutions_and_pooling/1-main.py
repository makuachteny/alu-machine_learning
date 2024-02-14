#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_same = __import__(
    '1-convolve_grayscale_same').convolve_grayscale_same


if __name__ == "__main__":
    dataset = np.load(
        "C:\\Users\\Lenovo\\Documents\\2. SE\\ALU\\alu-machine_learning\\math\\convolutions_and_pooling\\dataset\\mnist.npz")
    images = dataset['x_train']
    print(images.shape)
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale_same(images, kernel)
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    axes[0].imshow(images[0], cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(images_conv[0], cmap='gray')
    axes[1].set_title('Convolved Image')
    plt.tight_layout()
    plt.show()