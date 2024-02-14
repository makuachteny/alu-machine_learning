#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

convolve_grayscale_valid = __import__('0-convolve_grayscale_valid').convolve_grayscale_valid

if __name__ == "__main__":
    # Load the dataset
    dataset = np.load(
        "C:\\Users\\Lenovo\\Documents\\2. SE\\ALU\\alu-machine_learning\\math\\convolutions_and_pooling\\dataset\\mnist.npz")
    print(dataset.keys())  # Print out the keys present in the dataset

    # Extract the images from the dataset
    images = dataset['x_train']  
    
    # Extract the kernel from the dataset
    kernel =np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale_valid(images, kernel)
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2)
    
    # Plot the original images
    axes[0].imshow(images[0], cmap='gray')
    axes[0].set_title('Original Image')
    
    # Plot the convolved images
    axes[1].imshow(images_conv[0], cmap='gray')
    axes[1].set_title('Convolved Image')
    
    # Adjust the layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()
