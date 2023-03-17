import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v3 as imageio
import matplotlib.pyplot as plt
import cv2


img = imageio.imread("C:/Sparse_Dictionary_Learning/images/image1.jpg")

# Convert the image to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Add Gaussian noise to the image
mean = 0
variance = 100
sigma = np.sqrt(variance)
noisy_img = img + np.random.normal(mean, sigma, img.shape)

# Rescale the noisy image to the range [0, 255]
noisy_img = np.interp(noisy_img, (noisy_img.min(), noisy_img.max()), (0, 255)).astype(np.uint8)

D = np.load('learned_dictionary.npy')
Z = np.load('sparse_representation.npy')

# Normalize the noisy image
X_mean = np.mean(noisy_img, axis=0)
X_std = np.std(noisy_img, axis=0)
X_normalized = (noisy_img - X_mean) / X_std


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

axs[0].imshow(img, cmap='gray')
axs[0].axis('off')
axs[0].set_title('Original Image')

axs[1].imshow(noisy_img, cmap='gray')
axs[1].axis('off')
axs[1].set_title('Noisy Image')

plt.tight_layout()
plt.show()