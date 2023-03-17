
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v3 as imageio
import matplotlib.pyplot as plt
import cv2

images = []
for filename in os.listdir("C:/Sparse_Dictionary_Learning/images/"):
    img = imageio.imread(os.path.join("images", filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(img)

X = np.column_stack([img.flatten() for img in images])


X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_normalized = (X - X_mean) / X_std


# Initialize a dictionary of random atoms
n_atoms = 5
D_init = np.random.randn(X.shape[0], n_atoms)


D = D_init
Z = np.dot(np.linalg.pinv(D), X_normalized)

# Update the dictionary atoms
for j in range(n_atoms):
    Ij = np.where(Z[j, :] != 0)[0]
    if len(Ij) == 0:
        continue
    Dj = D[:, j]
    Rj = X_normalized - np.dot(D, Z) + np.outer(Dj, Z[j, :])
    D[:, j] = Rj[:, Ij].dot(Z[j, Ij].T)
    D[:, j] /= np.linalg.norm(D[:, j])

# Compute the reconstruction error and check for convergence
X_hat = np.dot(D, Z)
error = np.linalg.norm(X_normalized - X_hat)

# Save the learned dictionary and sparse representation as .npy files
np.save('learned_dictionary.npy', D)
np.save('sparse_representation.npy', Z)


# Reconstruct the images using the learned dictionary
X_reconstructed = np.dot(D, Z)

# Reverse the normalization
X_reconstructed = X_reconstructed * X_std + X_mean

# Reshape the columns of X_reconstructed into images
reconstructed_images = [X_reconstructed[:, i].reshape(images[0].shape) for i in range(X_reconstructed.shape[1])]


fig, axs = plt.subplots(nrows=len(images), ncols=2, figsize=(5, 10))

print(img.shape)
print(len(images))
print(len(reconstructed_images))

for i, (img, reconstructed_img) in enumerate(zip(images, reconstructed_images)):
    axs[i, 0].imshow(img, cmap='gray')
    axs[i, 0].axis('off')
    axs[i, 0].set_title('Original')
    axs[i, 1].imshow(reconstructed_img, cmap='gray')
    axs[i, 1].axis('off')
    axs[i, 1].set_title('Reconstructed, atoms:' + str(n_atoms))

plt.tight_layout()
plt.show()



