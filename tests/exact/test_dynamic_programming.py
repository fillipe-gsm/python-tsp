import pytest

from python_tsp.exact import solve_tsp_dynamic_programming
from tests.data import (
    distance_matrix1,
    distance_matrix2,
    distance_matrix3,
    optimal_permutation1,
    optimal_permutation2,
    optimal_permutation3,
    optimal_distance1,
    optimal_distance2,
    optimal_distance3,
)


@pytest.mark.parametrize(
    "distance_matrix", [distance_matrix1, distance_matrix2, distance_matrix3]
)
def test_solution_has_all_nodes(distance_matrix):
    """Check if the solution has all input nodes in any order"""

    permutation, _ = solve_tsp_dynamic_programming(distance_matrix)

    num_nodes = distance_matrix.shape[0]
    assert len(permutation) == num_nodes
    assert set(permutation) == set(range(num_nodes))


@pytest.mark.parametrize(
    "distance_matrix, expected_permutation, expected_distance",
    [
        (distance_matrix1, optimal_permutation1, optimal_distance1),
        (distance_matrix2, optimal_permutation2, optimal_distance2),
        (distance_matrix3, optimal_permutation3, optimal_distance3),
    ],
)
def test_solution_is_optimal(
    distance_matrix, expected_permutation, expected_distance
):
    """This exact method should return an optimal solution"""
    permutation, distance = solve_tsp_dynamic_programming(distance_matrix)

    assert permutation == expected_permutation
    assert distance == expected_distance
