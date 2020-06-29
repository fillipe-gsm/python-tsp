import pytest
import numpy as np

from python_tsp.local_search import solve_tsp_two_opt


distance_matrix = np.array([
    [0, 4, 8, 2, 1, 5],
    [4, 0, 3, 3, 3, 6],
    [8, 3, 0, 7, 7, 8],
    [2, 3, 7, 0, 0, 6],
    [1, 3, 7, 0, 0, 5],
    [5, 6, 8, 6, 5, 0],
])


@pytest.fixture
def import_two_opt_swap():
    """Fixture importing the auxiliar _two_opt_swap"""
    from python_tsp.local_search.two_opt import _two_opt_swap
    return _two_opt_swap


class Test2OptLocalSearch:
    def test_2opt_swap_works_for_valid_input(self, import_two_opt_swap):
        """The 2-opt swap works as expected if i <= j"""

        permutation = [0, 1, 2, 3, 4, 5, 6]
        expected_response = [0, 1, 5, 4, 3, 2, 6]

        response = import_two_opt_swap(permutation, 2, 6)

        assert response == expected_response

    def test_2opt_swap_raises_for_invalid_input(self, import_two_opt_swap):
        """The 2-opt swap raises an exception if i > j"""

        permutation = [0, 1, 2, 3, 4, 5, 6]
        with pytest.raises(ValueError):
            import_two_opt_swap(permutation, 5, 2)

    def test_solution_has_all_nodes_closed_problem(self, distance_matrix):
        """The solution returned has all nodes in the closed version"""

        permutation, _ = solve_tsp_two_opt(distance_matrix)
