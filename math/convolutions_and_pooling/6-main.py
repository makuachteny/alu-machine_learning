#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
pool = __import__('6-pool').pool


if __name__ == '__main__':

    dataset = np.load(
        'C:\\Users\\Lenovo\\Documents\\2. SE\ALU\\alu-machine_learning\math\convolutions_and_pooling\\dataset\\animals_1.npz')
    images = dataset['data']
    print(images.shape)
    images_pool = pool(images, (2, 2), (2, 2), mode='avg')
    print(images_pool.shape)

fig, axes = plt.subplots(1, 2)

axes[0].imshow(images[0])
axes[0].set_title('Original Image')

axes[1].imshow(images_pool[0])
axes[1].set_title('Pooled Image')

plt.show()
