import pytest

from python_tsp.local_search import solve_tsp_two_opt
from python_tsp.utils import compute_permutation_distance
from python_tsp.local_search.two_opt import (
    _initial_solution, _neighborhood_search, _two_opt_swap
)  # import auxiliar functions for tests
from .data import (
    distance_matrix1, distance_matrix2, distance_matrix3,
    optimal_permutation1, optimal_permutation2, optimal_permutation3,
    optimal_distance1, optimal_distance2, optimal_distance3
)


class TestImportModule:
    def test_can_import_local_serach_solvers(self):
        """
        Check if the main functions can be imported from the `local_search`
        package
        """
        from python_tsp.local_search import (  # noqa
            solve_tsp_two_opt
        )


class Test2OptLocalSearch:
    def test_2opt_swap_works_for_valid_input(self):
        """The 2-opt swap works as expected if i <= j"""

        permutation = [0, 1, 2, 3, 4, 5, 6]
        expected_response = [0, 1, 5, 4, 3, 2, 6]

        response = _two_opt_swap(permutation, 2, 6)

        assert response == expected_response

    def test_2opt_swap_raises_for_invalid_input(self):
        """The 2-opt swap raises an exception if i > j"""

        permutation = [0, 1, 2, 3, 4, 5, 6]
        with pytest.raises(ValueError):
            _two_opt_swap(permutation, 5, 2)

    @pytest.mark.parametrize(
        "distance_matrix, expected_distance",
        [
            (distance_matrix1, 28),
            (distance_matrix2, 26),
            (distance_matrix3, 20)
        ]
    )
    def test_setup_return_same_initial_solution(
        self, distance_matrix, expected_distance
    ):
        """
        The _initial_solution outputs the same input if provided, together with
        its distance
        """

        initial_permutation = [0, 2, 3, 1, 4]
        x, fx = _initial_solution(distance_matrix, initial_permutation)

        assert x == initial_permutation
        assert fx == expected_distance

    @pytest.mark.parametrize(
        "distance_matrix",
        [distance_matrix1, distance_matrix2, distance_matrix3]
    )
    def test_setup_return_random_valid_solution(self, distance_matrix):
        """
        The _initial_solution outputs a random valid permutation if no
        initial solution is provided
        """

        x, fx = _initial_solution(distance_matrix)

        assert set(x) == set(range(distance_matrix.shape[0]))
        assert fx

    @pytest.mark.parametrize(
        "distance_matrix",
        [distance_matrix1, distance_matrix2, distance_matrix3]
    )
    def test_neighborhood_search_returns_better_neighbor(
        self, distance_matrix
    ):
        """
        If there is room for improvement, a better neighbor is returned.
        Here, we choose purposely a permutation that can be improved.
        """
        x = [0, 4, 2, 3, 1]
        fx = compute_permutation_distance(distance_matrix, x)
        flag, xn, fn = _neighborhood_search(distance_matrix, x, fx)

        assert flag
        assert fn < fx

    @pytest.mark.parametrize(
        "distance_matrix, optimal_permutation, optimal_distance",
        [
            (distance_matrix1, optimal_permutation1, optimal_distance1),
            (distance_matrix2, optimal_permutation2, optimal_distance2),
            (distance_matrix3, optimal_permutation3, optimal_distance3),
        ]
    )
    def test_neighborhood_search_returns_equal_neighbor(
        self, distance_matrix, optimal_permutation, optimal_distance
    ):
        """
        If there is no room for improvement, the same neighbor is returned.
        Here, we choose purposely the optimal solution of each problem
        """
        x = optimal_permutation
        fx = optimal_distance
        flag, xn, fn = _neighborhood_search(distance_matrix, x, fx)

        assert not flag
        assert xn == x
        assert fn == fx

    @pytest.mark.parametrize(
        "distance_matrix",
        [distance_matrix1, distance_matrix2, distance_matrix3]
    )
    def test_solution_has_all_nodes(self, distance_matrix):
        """Check if the solution has all input nodes in any order
        """

        permutation, _ = solve_tsp_two_opt(distance_matrix)

        n = distance_matrix.shape[0]
        assert len(permutation) == n
        assert set(permutation) == set(range(n))
