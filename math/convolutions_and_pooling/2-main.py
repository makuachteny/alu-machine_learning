#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_padding = __import__('2-convolve_grayscale_padding').convolve_grayscale_padding


if __name__ == '__main__':

    dataset = np.load(
        'C:\\Users\\Lenovo\\Documents\\2. SE\\ALU\\alu-machine_learning\\math\\convolutions_and_pooling\\dataset\\mnist.npz')
    images = dataset['x_train']
    print(images.shape)
    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale_padding(images, kernel, (2, 4))
    print(images_conv.shape)
    
    # Create a subplot with 1 row and two columns
    fig, axes = plt.subplots(1, 2)
    
    # Display the original image
    axes[0].imshow(images[0], cmap='gray')
    axes[0].set_title('Original Image')
    
    # Display the convolved image
    axes[1].imshow(images_conv[0], cmap='gray')
    axes[1].set_title('Convolved Image')
    
    # Adjust layout to avoid overlaps
    plt.tight_layout()
    
    # Display the plot
    plt.show()
    