import pytest

from python_tsp.utils import compute_permutation_distance
from .data import (
    distance_matrix1, distance_matrix2, distance_matrix3,
)


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
        distance = compute_permutation_distance(distance_matrix, permutation)

        assert distance == expected_distance

    @pytest.mark.parametrize(
        "distance_matrix, expected_distance",
        [
            (distance_matrix1, 28),
            (distance_matrix2, 26),
            (distance_matrix3, 20)
        ]
    )
    def test_return_correct_permutation_distance_initial_node_not_zero(
        self, distance_matrix, expected_distance
    ):
        """
        Check if the correct distance is returned when the first node is not 0
        """
        # Same permutation as before but starting at 3
        permutation = [3, 1, 4, 0, 2]
        distance = compute_permutation_distance(distance_matrix, permutation)

        assert distance == expected_distance
