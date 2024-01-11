import numpy as np
import pytest

from python_tsp.exact import solve_tsp_branch_and_bound
from tests.data import (
    distance_matrix1,
    distance_matrix2,
    distance_matrix3,
    optimal_distance1,
    optimal_distance2,
    optimal_distance3,
)


@pytest.mark.parametrize(
    "distance_matrix", [distance_matrix1, distance_matrix2, distance_matrix3]
)
def test_solution_has_all_nodes(distance_matrix):
    """
    Test the `solve_tsp_branch_and_bound` function for the presence of all
    input nodes.

    Verifies if the solution contains all input nodes in any order.

    The function is tested with three different distance matrices.

    Verifications:
        - The length of the permutation should be equal to the number of
          nodes.
        - The set of nodes in the permutation should be equal to the set of all
          nodes.
    """
    permutation, _ = solve_tsp_branch_and_bound(distance_matrix)

    num_nodes = distance_matrix.shape[0]
    assert len(permutation) == num_nodes
    assert set(permutation) == set(range(num_nodes))


@pytest.mark.parametrize(
    "distance_matrix, expected_distance",
    [
        (distance_matrix1, optimal_distance1),
        (distance_matrix2, optimal_distance2),
        (distance_matrix3, optimal_distance3),
    ],
)
def test_solution_is_optimal(distance_matrix, expected_distance):
    """
    Test the `solve_tsp_branch_and_bound` function for optimality.

    Verifies if the exact method returns an optimal solution.

    The function is tested with three different distance matrices and their
    corresponding optimal distances.

    Verifications:
        - The distance returned by the function should be equal to the
          expected optimal distance.
    """
    _, distance = solve_tsp_branch_and_bound(distance_matrix)

    assert distance == expected_distance


def test_solver_on_an_unfeasible_problem():
    """
    Test the `solve_tsp_branch_and_bound` function on an unfeasible
    problem.

    Verifies the behavior of the function when provided with an unfeasible
    distance matrix.

    Verifications:
        - The permutation of nodes in the solution should be empty.
        - The distance of the solution should be positive infinity.
    """
    inf = float("inf")
    distance_matrix = np.array(
        [
            [inf, 10, 15, 20, inf],
            [inf, inf, 12, inf, 25],
            [inf, inf, inf, 8, 18],
            [inf, inf, inf, inf, inf],
            [inf, inf, inf, inf, inf],
        ]
    )
    permutation, distance = solve_tsp_branch_and_bound(distance_matrix)

    assert permutation == []
    assert distance == inf
