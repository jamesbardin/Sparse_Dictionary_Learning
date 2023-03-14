# 1. load required libraries and dependencies
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os

# 2. load the images form the specified folder and preprocess
# path to folder with images
path = 'C:/Sparse_Dictionary_Learning/images/'

# load images in folder
image_list = []
for filename in os.listdir(path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img = imageio.imread(os.path.join(path, filename), as_gray = True)
        image_list.append(img)
        
# Convert the list of images to a numpy array
images = np.asarray(image_list)

# Preprocess the images (you can choose your own preprocessing steps here)
images = images.astype(np.float32) / 255.0  # Normalize the pixel values between 0 and 1
images -= np.mean(images, axis=0)  # Subtract the mean image to center the data


# 3. initialize the dicitonary with random vals or predefined basis fuctions
# set number of atoms and patch size
num_atoms = 64
patch_size = (8,8)

# Initialize the dictionary with random values
dictionary = np.random.randn(patch_size[0] * patch_size[1], num_atoms)

# Normalize the columns of the dictionary
dictionary /= np.linalg.norm(dictionary, axis=0)


# Alternatively, we could initialize the dictionary with predefined basis functions
# (such as wavelets, fourier basis, or curvelets)
# dictionary = build_basis(patch_size, num_atoms)



# 4. implement optimization algorithm (K-SVD) to update the dictionary and sparse codes iteratively

# Set the number of iterations and the sparsity level
num_iterations = 5
sparsity_level = 5

# Define the k-SVD function
def ksvd(data, dictionary, sparsity_level, num_iterations):
    for i in range(num_iterations):
        # Sparse coding step
        coefficients = sparse_encode(data, dictionary, sparsity_level)
        
        # Dictionary update step
        dictionary = dictionary_update(data, dictionary, coefficients)
        
    return dictionary, coefficients

# Define the sparse encoding function
def sparse_encode(data, dictionary, sparsity_level):
    coefficients = np.zeros((dictionary.shape[1], data.shape[0]))
    for i in range(data.shape[0]):
        x = data[i]
        active_set = []
        residual = x.copy()
        for j in range(sparsity_level):
            # Find the index of the most correlated atom
            corr = np.dot(dictionary.T, residual)
            idx = np.argmax(np.abs(corr))
            active_set.append(idx)
            
            # Solve the least squares problem to update the coefficients
            A = dictionary[:, active_set]
            y = x - np.dot(A, coefficients[active_set, i])
            coef = np.linalg.lstsq(A, y, rcond=None)[0]
            
            # Update the coefficients
            coefficients[active_set, i] = coef.reshape(-1)
            residual = x - np.dot(A, coef)
            
            # Remove the atom if its correlation with the residual is low
            if np.linalg.norm(residual) < 1e-6:
                break
            corr[idx] = -np.inf
        
    return coefficients

# Define the dictionary update function
def dictionary_update(data, dictionary, coefficients):
    for j in range(dictionary.shape[1]):
        # Find the indices of the data points that use the j-th atom
        idx = np.nonzero(coefficients[j, :])[0]
        
        if len(idx) > 0:
            # Compute the residual for the j-th atom
            r = data[idx].T - np.dot(dictionary, coefficients[:, idx]) + np.outer(dictionary[:, j], coefficients[j, idx])

            print(r.shape)
            
            # Find the SVD of the residual matrix
            u, s, v = np.linalg.svd(r, full_matrices=False)
            
            # Update the j-th atom
            dictionary[:, j] = u[:, 0]
            coefficients[j, idx] = s[0] * v[0, :]
            
    # Normalize the columns of the dictionary
    dictionary /= np.linalg.norm(dictionary, axis=0)
    
    return dictionary

# Run the k-SVD algorithm on the images
learned_dictionary, coefficients = ksvd(images.reshape(-1, patch_size[0]*patch_size[1]), dictionary, sparsity_level, num_iterations)



# 5. reconstruct using the learned dictionary and varying num of atoms



# display and save the reconstructed images with the different number of atoms

