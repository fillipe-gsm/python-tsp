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
    """Check if the solution has all input nodes in any order"""
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
    """This exact method should return an optimal solution"""
    _, distance = solve_tsp_branch_and_bound(distance_matrix)

    assert distance == expected_distance


def test_solver_on_an_unfeasible_problem():
    inf = np.iinfo(int).max
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
    assert distance == 0
