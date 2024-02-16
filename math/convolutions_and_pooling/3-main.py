import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale = __import__('3-convolve_grayscale').convolve_grayscale


if __name__ == '__main__':

    dataset = np.load(
        'C:\\Users\\Lenovo\\Documents\\2. SE\\ALU\\alu-machine_learning\\math\\convolutions_and_pooling\\dataset\\mnist.npz')
    images = dataset['x_train']
    print(images.shape)
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale(
        images, kernel, padding='valid', stride=(2, 2))
    print(images_conv.shape)
    
    # Display both the original and the convolved on the same figure
    fig, axes = plt.subplots(1, 2)
    
    # Display the original image
    axes[0].imshow(images[0], cmap='gray')
    axes[0].set_title('Original Image')
    
    # Display the convolved image
    axes[1].imshow(images_conv[0], cmap='gray')
    axes[1].set_title('Convolved Image')

    # plt.imshow(images[0], cmap='gray')
    # plt.show()
    # plt.imshow(images_conv[0], cmap='gray')
    plt.show()
