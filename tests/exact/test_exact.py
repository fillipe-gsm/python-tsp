import pytest

from python_tsp.exact import (
    solve_tsp_brute_force, solve_tsp_dynamic_programming
)
from .data import (
    distance_matrix1, distance_matrix2, distance_matrix3,
    optimal_permutation1, optimal_permutation2, optimal_permutation3,
    optimal_distance1, optimal_distance2, optimal_distance3
)


class TestImportModule:
    def test_can_import_exact_solvers(self):
        """
        Check if the main functions can be imported from the `exact` package
        """
        from python_tsp.exact import (  # noqa
            solve_tsp_brute_force, solve_tsp_dynamic_programming
        )


class TestBruteForceAlgorithm:
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
        [distance_matrix1, distance_matrix2, distance_matrix3]
    )
    def test_solution_has_all_nodes(self, distance_matrix):
        """Check if the solution has all input nodes in any order
        """

        permutation, _ = solve_tsp_dynamic_programming(distance_matrix)

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
        permutation, distance = solve_tsp_dynamic_programming(distance_matrix)

        assert permutation == expected_permutation
        assert distance == expected_distance
