#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_channels = __import__('4-convolve_channels').convolve_channels


if __name__ == '__main__':

    dataset = np.load(
        'C:\\Users\\Lenovo\\Documents\\2. SE\ALU\\alu-machine_learning\math\convolutions_and_pooling\\dataset\\animals_1.npz')

    images = dataset['data']
    print(images.shape)
    kernel = np.array([[[0, 0, 0], [-1, -1, -1], [0, 0, 0]], [[-1, -1, -1],
                      [5, 5, 5], [-1, -1, -1]], [[0, 0, 0], [-1, -1, -1], [0, 0, 0]]])
    images_conv = convolve_channels(images, kernel, padding='valid')
    print(images_conv.shape)

    fig, axes = plt.subplots(1, 2)
    
    axes[0].imshow(images[0])
    axes[0].set_title('Original Image')    
    
    axes[1].imshow(images_conv[0], cmap='viridis')
    axes[1].set_title('Convolved Image')
    
    plt.show()
