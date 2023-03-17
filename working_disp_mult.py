import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v3 as imageio
import cv2
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Sparse dictionary learning')
parser.add_argument('images_folder', metavar='images_folder', type=str,
                    help='Path to the folder containing images')
parser.add_argument('--n_atoms_list', metavar='n_atoms_list', type=int, nargs='+', default=[1, 2, 5, 10, 20, 50],
                    help='List of numbers of atoms to use for reconstruction')
args = parser.parse_args()

# Read img from a folder into a list, convert to grayscale
def read_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        # Read in image with imageio library
        img = imageio.imread(os.path.join(folder_path, filename))
        # Convert image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img)
    return images

# Normalize data to zero mean and unit variance
def normalize_data(data):
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    normalized_data = (data - data_mean) / data_std
    return normalized_data, data_mean, data_std

# Update dictionary atoms using sparse coding
def update_dictionary(D, Z, X_normalized):
    # Loop through each column of dict. matrix
    for j in range(D.shape[1]):
        # Find indices only of nonzero elements
        Ij = np.where(Z[j, :] != 0)[0]
        if len(Ij) == 0:
            continue
        Dj = D[:, j]
        # Compute residual error matrix
        Rj = X_normalized - np.dot(D, Z) + np.outer(Dj, Z[j, :])
        # Update jth atom w/ residual error and nonzero elements
        D[:, j] = Rj[:, Ij].dot(Z[j, Ij].T)
        # Normalize
        D[:, j] /= np.linalg.norm(D[:, j])
    return D

# Reconstruct images using the learned dictionary
def reconstruct_images(D, Z, data_mean, data_std, n_atoms):
    reconstructed_data = np.dot(D[:, :n_atoms], Z[:n_atoms, :])
    reconstructed_data = reconstructed_data * data_std + data_mean
    # reshape into a list of images
    reconstructed_images = [reconstructed_data[:, i].reshape(images[0].shape) for i in range(reconstructed_data.shape[1])]
    return reconstructed_images


# Load images and convert them to grayscale
images_folder = args.images_folder
images = read_images(images_folder)


# Normalize the data
X, X_mean, X_std = normalize_data(np.column_stack([img.flatten() for img in images]))

# Initialize a dictionary of random atoms
n_atoms_list = args.n_atoms_list
D_init = np.random.randn(X.shape[0], n_atoms_list[-1])
D = D_init

fig, axs = plt.subplots(nrows=len(images), ncols=2+len(n_atoms_list) - 1, figsize=(12, 10))

# Loop over the images
for i, img in enumerate(images):
    # Display the original
    axs[i, 0].imshow(img, cmap='gray')
    axs[i, 0].axis('off')
    axs[i, 0].set_title('Original')

    # Loop over different numbers of atoms and display
    for j, n_atoms in enumerate(n_atoms_list):
        # Sparse coding and dictionary update
        Z = np.dot(np.linalg.pinv(D[:, :n_atoms]), X)
        D[:, :n_atoms] = update_dictionary(D[:, :n_atoms], Z, X)

        # Reconstruct the images using the learned dictionary
        reconstructed_images = reconstruct_images(D, Z, X_mean, X_std, n_atoms)

        # Plot the reconstructed images
        axs[i, j+1].imshow(reconstructed_images[i], cmap='gray')
        axs[i, j+1].axis('off')
        axs[i, j+1].set_title('Reconst. atoms={}'.format(n_atoms))

plt.tight_layout()
plt.show()
