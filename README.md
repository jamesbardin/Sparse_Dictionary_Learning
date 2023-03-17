# Sparse Dictionary Learning
#### James Bardin

To install the running environment navigate to the directory with the yaml and run:

    '''conda env create -f .\environment.yml -n sparse_dictionary_learning'''

usage: python .\working_disp_mult.py [--n_atoms_list n_atoms_list [n_atoms_list ...]] images_folder

(the default n_atoms_list is [1, 2, 5, 10, 20, 50])

To run on my machine, an example command to display for atoms=1,2,3 is:

    '''python .\working_disp_mult.py 'C:\Sparse_Dictionary_Learning\images\' --n_atoms_list 1 2 3'''

