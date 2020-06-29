import numpy as np
import pytest

from python_tsp.exact import (
    solve_tsp_brute_force, solve_tsp_dynamic_programming
)


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


class TestImportModule:
    def test_can_import_exact_solvers(self):
        """
        Check if the main functions can be imported from the `exact` package
        """
        from python_tsp.exact import (  # noqa
            solve_tsp_brute_force, solve_tsp_dynamic_programming
        )

    def test_cannot_import_auxiliar_function(self):
        """
        Check if auxiliar function inside modules are not importable directly
        from `exact` package
        """
        with pytest.raises(ImportError):
            from python_tsp.exact import _permutation_distance  # noqa


class TestBruteForceAlgorithm:
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
        """Check if the correct distance is returned for a given permutation
        This function tests a subfunction inside the module, so it is imported
        explicitly in this case
        """
        from python_tsp.exact.brute_force import _permutation_distance

        # Get a random permutation (node 0 is automatically inserted in the
        # code)
        permutation = (2, 3, 1, 4)
        distance = _permutation_distance(
            permutation, distance_matrix
        )

        assert distance == expected_distance

    @pytest.mark.parametrize(
        "distance_matrix",
        [distance_matrix1, distance_matrix2, distance_matrix3]
    )
    def test_solution_has_all_nodes(self, distance_matrix):
        """Check if the solution has all input nodes in any order
        """

        permutation, _ = solve_tsp_brute_force(distance_matrix)

        n = distance_matrix.shape[0]
        assert len(permutation) == n
        assert set(permutation) == set(range(n))

    @pytest.mark.parametrize(
        "distance_matrix, expected_permutation, expected_distance",
        [
            (distance_matrix1, optimal_permutation1, optimal_distance1),
            (distance_matrix2, optimal_permutation2, optimal_distance2),
            (distance_matrix3, optimal_permutation3, optimal_distance3),
        ]
    )
    def test_solution_is_optimal(
        self, distance_matrix, expected_permutation, expected_distance
    ):
        """This exact method should return an optimal solution"""
        permutation, distance = solve_tsp_brute_force(distance_matrix)

        assert permutation == expected_permutation
        assert distance == expected_distance


class TestDynamicProgrammingAlgorithm:
    @pytest.mark.parametrize(
        "distance_matrix",
        [distance_matrix1, distance_matrix2]
    )
    def test_solution_has_all_nodes_closed_problem(self, distance_matrix):
        """Check if the solution has all input nodes in the closed version
        For an input with n nodes, the solution must have lenght n + 1, with
        the starting node being repeated, and all nodes from 0 to n.
        """

        permutation, _ = solve_tsp_dynamic_programming(
            distance_matrix
        )

        n = distance_matrix.shape[0]
        assert len(permutation) == n + 1
        assert set(permutation) == set(range(n))

    @pytest.mark.parametrize(
        "distance_matrix",
        [distance_matrix1, distance_matrix2]
    )
    def test_solution_has_all_nodes_open_problem(self, distance_matrix):
        """Check if the solution has all input nodes in the open version
        For an input with n nodes, the solution must have lenght n, plus all
        nodes from 0 to n.
        """
        permutation, _ = solve_tsp_dynamic_programming(
            distance_matrix, open_tsp=True
        )

        n = distance_matrix.shape[0]
        assert len(permutation) == n
        assert set(permutation) == set(range(n))

    @pytest.mark.parametrize(
        "distance_matrix, expected_permutation, expected_distance",
        [
            (distance_matrix1, [0, 1, 2, 0], 11),
            (distance_matrix2, [0, 2, 1, 0], 7)
        ]
    )
    def test_solution_is_optimal_closed_problem(
        self, distance_matrix, expected_permutation, expected_distance
    ):
        """This exact method should return an optimal solution"""
        permutation, distance = solve_tsp_dynamic_programming(
            distance_matrix
        )

        assert permutation == expected_permutation
        assert distance == expected_distance

    @pytest.mark.parametrize(
        "distance_matrix, expected_permutation, expected_distance",
        [
            (distance_matrix1, [0, 1, 2], 7),
            (distance_matrix2, [0, 2, 1], 6)
        ]
    )
    def test_solution_is_optimal_open_problem(
        self, distance_matrix, expected_permutation, expected_distance
    ):
        """
        This exact method should return an optimal solution in the open case
        """
        permutation, distance = solve_tsp_dynamic_programming(
            distance_matrix, open_tsp=True
        )

        assert permutation == expected_permutation
        assert distance == expected_distance
