import numpy as np
import pytest

from python_tsp.utils import compute_permutation_distance

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


class TestComputePermutationDistance:
    @pytest.mark.parametrize(
        "distance_matrix, expected_distance",
        [
            (distance_matrix1, 28),
            (distance_matrix2, 26),
            (distance_matrix3, 20)
        ]
    )
    def test_return_correct_permutation_distance(
        self, distance_matrix, expected_distance
    ):
        """
        Check if the correct distance is returned for a given permutation and
        a given distance matrix
        """
        permutation = [0, 2, 3, 1, 4]
        distance = compute_permutation_distance(permutation, distance_matrix)

        assert distance == expected_distance
