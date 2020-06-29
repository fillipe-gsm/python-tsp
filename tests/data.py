"""
Module with some variables used in most test files, but too simple to be
considered fixtures
"""
import numpy as np


# Symmetric distance matrix
distance_matrix1 = np.array([
    [0, 2, 4, 6, 8],
    [2, 0, 3, 5, 7],
    [4, 6, 0, 4, 6],
    [6, 5, 4, 0, 7],
    [8, 5, 6, 7, 0],
])
optimal_permutation1 = [0, 2, 3, 4, 1]
optimal_distance1 = 22

# Unsymmetric distance matrix
distance_matrix2 = np.array([
    [0, 2, 4, 6, 8],
    [3, 0, 3, 5, 7],
    [4, 7, 0, 4, 6],
    [5, 5, 3, 0, 7],
    [6, 3, 4, 5, 0],
])
optimal_permutation2 = [0, 1, 2, 4, 3]
optimal_distance2 = 21

# Open problem (the returning cost is 0)
distance_matrix3 = np.array([
    [0, 2, 4, 6, 8],
    [0, 0, 3, 5, 7],
    [0, 6, 0, 4, 6],
    [0, 5, 4, 0, 7],
    [0, 5, 6, 7, 0],
])
optimal_permutation3 = [0, 1, 2, 3, 4]
optimal_distance3 = 16
