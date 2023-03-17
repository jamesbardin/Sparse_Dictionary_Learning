import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v3 as imageio
# from PIL import Image
import cv2
from sklearn.decomposition import DictionaryLearning

images = []
for filename in os.listdir("C:/Sparse_Dictionary_Learning/images/"):
    img = imageio.imread(os.path.join("images", filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(img)

# Convert  images into a matrix
X = np.column_stack([img.flatten() for img in images])

# Normalize the columns of X
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_normalized = (X - X_mean) / X_std


n_atoms = 1
n_features = X.shape[0]
D_init = np.random.randn(X.shape[0], n_atoms)

n_iter = 1
dl = DictionaryLearning(n_components=n_atoms, dict_init=D_init, transform_algorithm='omp', max_iter=n_iter, verbose=1)
D_learned = dl.fit(X_normalized).components_
Z = dl.transform(X_normalized)

# Reconstruct the images using the learned dictionary and sparse representation coefficients
X_reconstructed = np.dot(D_learned.T, Z.T).T

# Denormalize the columns of X and X_reconstructed
X_denormalized = X_std * X_normalized + X_mean
X_reconstructed_denormalized = X_std * X_reconstructed + X_mean

# Compute the residual error between X and X_reconstructed
residual_error = np.linalg.norm(X_denormalized - X_reconstructed_denormalized) / np.linalg.norm(X_denormalized)

print("Residual error:", residual_error)


# Reshape the first two images and their reconstructions
img1 = X_denormalized[:, 0].reshape(images[0].shape)
img1_reconstructed = X_reconstructed_denormalized[:, 0].reshape(images[0].shape)
img2 = X_denormalized[:, 1].reshape(images[1].shape)
img2_reconstructed = X_reconstructed_denormalized[:, 1].reshape(images[1].shape)

# Display the images side-by-side
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].imshow(img1, cmap='gray')
axs[0, 0].set_title('Original Image 1')
axs[0, 1].imshow(img1_reconstructed, cmap='gray')
axs[0, 1].set_title('Reconstructed Image 1')
axs[1, 0].imshow(img2, cmap='gray')
axs[1, 0].set_title('Original Image 2')
axs[1, 1].imshow(img2_reconstructed, cmap='gray')
axs[1, 1].set_title('Reconstructed Image 2')
plt.show()