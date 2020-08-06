import pytest

from python_tsp.heuristics import local_search
from python_tsp.utils import compute_permutation_distance
from .data import (
    distance_matrix1, distance_matrix2, distance_matrix3,
    optimal_permutation1, optimal_permutation2, optimal_permutation3,
    optimal_distance1, optimal_distance2, optimal_distance3
)


class TestImportModule:
    def test_can_import_local_search_solver(self):
        """
        Check if the main function can be imported from the `heuristics`
        package
        """
        from python_tsp.heuristics import (  # noqa
            solve_local_search
        )


class TestPerturbationSchemes:
    x = [0, 1, 2, 3, 4]

    def test_ps1_returns_correct_num_neighbors(self):
        """PS1 has n - 1 swaps.
        But since we fix the first element as origin, it leads to n - 2 = 3 in
        this case.
        """

        all_neighbors = list(local_search.ps1_gen(self.x))
        assert len(all_neighbors) == 3

    def test_ps2_returns_correct_num_neighbors(self):
        """PS2 has n * (n - 1) / 2 swaps.
        But since we fix the first element as origin, it leads to
        (n - 1) * (n - 2) / 2 = 6 in this case.
        """

        all_neighbors = list(local_search.ps2_gen(self.x))
        assert len(all_neighbors) == 6

    def test_ps3_returns_correct_num_neighbors(self):
        """PS3 has n * (n - 1) elements.
        But since we fix the first element as origin, it leads to
        (n - 1) * (n - 2) = 12 in this case.
        """

        all_neighbors = list(local_search.ps3_gen(self.x))
        assert len(all_neighbors) == 12


class TestLocalSearch:
    @pytest.mark.parametrize(
        "distance_matrix, expected_distance",
        [
            (distance_matrix1, 28),
            (distance_matrix2, 26),
            (distance_matrix3, 20)
        ]
    )
    def test_setup_return_same_setup(
        self, distance_matrix, expected_distance
    ):
        """
        The _setup outputs the same input if provided, together with
        its objective value
        """

        x0 = [0, 2, 3, 1, 4]
        x, fx = local_search._setup(distance_matrix, x0)

        assert x == x0
        assert fx == expected_distance

    @pytest.mark.parametrize(
        "distance_matrix",
        [distance_matrix1, distance_matrix2, distance_matrix3]
    )
    def test_setup_return_random_valid_solution(self, distance_matrix):
        """
        The _setup outputs a random valid permutation if no
        initial solution is provided
        """

        x, fx = local_search._setup(distance_matrix)

        assert set(x) == set(range(distance_matrix.shape[0]))
        assert fx

    @pytest.mark.parametrize("scheme", ["ps1", "ps2", "ps3"])
    @pytest.mark.parametrize(
        "distance_matrix",
        [distance_matrix1, distance_matrix2, distance_matrix3]
    )
    def test_local_search_returns_better_neighbor(
        self, scheme, distance_matrix
    ):
        """
        If there is room for improvement, a better neighbor is returned.
        Here, we choose purposely a permutation that can be improved.
        """
        x = [0, 4, 2, 3, 1]
        fx = compute_permutation_distance(distance_matrix, x)

        xopt, fopt = local_search.solve_local_search(
            distance_matrix, x, perturbation_scheme=scheme
        )

        assert fopt < fx

    @pytest.mark.parametrize("scheme", ["ps1", "ps2", "ps3"])
    @pytest.mark.parametrize(
        "distance_matrix, optimal_permutation, optimal_distance",
        [
            (distance_matrix1, optimal_permutation1, optimal_distance1),
            (distance_matrix2, optimal_permutation2, optimal_distance2),
            (distance_matrix3, optimal_permutation3, optimal_distance3),
        ]
    )
    def test_local_search_returns_equal_optimal_solution(
        self, scheme, distance_matrix, optimal_permutation, optimal_distance
    ):
        """
        If there is no room for improvement, the same solution is returned.
        Here, we choose purposely the optimal solution of each problem
        """
        x = optimal_permutation
        fx = optimal_distance
        xopt, fopt = local_search.solve_local_search(
            distance_matrix, x, perturbation_scheme=scheme
        )

        assert xopt == x
        assert fopt == fx
