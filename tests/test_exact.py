import numpy as np
import pytest

from python_tsp import brute_force


symmetric_distance_matrix = np.array([
    [0, 2, 4],
    [2, 0, 5],
    [4, 5, 0]
])

unsymmetric_distance_matrix = np.array([
    [0, 2, 4],
    [1, 0, 6],
    [5, 2, 0],
])


class TestBruteForceAlgorithm:
    @pytest.mark.parametrize(
        "distance_matrix, expected_distance",
        [(symmetric_distance_matrix, 11), (unsymmetric_distance_matrix, 7)]
    )
    def test_return_correct_permutation_distance(
        self, distance_matrix, expected_distance
    ):
        """Check if the correct distance is returned for a given permutation
        For the permutation [0, 2, 1], the correct distances are:
            - 4 + 5 + 2 = 11, for the symmetric case
            - 4 + 2 + 1 = 7, for the unsymmetric case
        """
        permutation = (2, 1)  # node 0 is automatically inserted in the code
        distance = brute_force._permutation_distance(
            permutation, distance_matrix
        )

        assert distance == expected_distance

    @pytest.mark.parametrize(
        "distance_matrix",
        [symmetric_distance_matrix, unsymmetric_distance_matrix]
    )
    def test_solution_has_all_nodes_closed_problem(self, distance_matrix):
        """Check if the solution has all input nodes in the closed version
        For an input with n nodes, the solution must have lenght n + 1, with
        the starting node being repeated, and all nodes from 0 to n.
        """

        permutation, _ = brute_force.solve_tsp_brute_force(
            distance_matrix
        )

        n = distance_matrix.shape[0]
        assert len(permutation) == n + 1
        assert set(permutation) == set(range(n))

    @pytest.mark.parametrize(
        "distance_matrix",
        [symmetric_distance_matrix, unsymmetric_distance_matrix]
    )
    def test_solution_has_all_nodes_open_problem(self, distance_matrix):
        """Check if the solution has all input nodes in the open version
        For an input with n nodes, the solution must have lenght n, plus all
        nodes from 0 to n.
        """
        permutation, _ = brute_force.solve_tsp_brute_force(
            distance_matrix, open_tsp=True
        )

        n = distance_matrix.shape[0]
        assert len(permutation) == n
        assert set(permutation) == set(range(n))
