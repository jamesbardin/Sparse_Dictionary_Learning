# Sparse Dictionary Learning
#### James Bardin

To install the running environment navigate to the directory with the yaml and run:

conda env create -f .\environment.yml -n sparse_dictionary_learning

usage: python .\working_disp_mult.py [--n_atoms_list n_atoms_list [n_atoms_list ...]] images_folder

(the default n_atoms_list is [1, 2, 5, 10, 20, 50])

To run on my machine, an example command to display for atoms=1,2,3 is:

python .\working_disp_mult.py 'C:\Sparse_Dictionary_Learning\images\' --n_atoms_list 1 2 3


Basic steps followed:
- Load the input images.
- Convert the images to grayscale if necessary, using a library such as cv2.
- Convert the image data to a matrix form, where each column represents a flattened image.
- Normalize the image data by subtracting the mean and dividing by the standard deviation.
- Initialize a dictionary of atoms, either randomly or using a predefined set of basis functions.
- Learn the dictionary atoms and sparse representation coefficients.
- Repeat with different numbers of dictionary atoms
