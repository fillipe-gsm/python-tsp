import pytest

from python_tsp.utils import setup_initial_solution
from tests.data import (
    distance_matrix1,
    distance_matrix2,
    distance_matrix3,
)


@pytest.mark.parametrize(
    "distance_matrix, expected_distance",
    [(distance_matrix1, 28), (distance_matrix2, 26), (distance_matrix3, 20)],
)
def test_setup_return_same_setup(distance_matrix, expected_distance):
    """
    The setup outputs the same input if provided, together with its
    objective value
    """

    x0 = [0, 2, 3, 1, 4]
    x, fx = setup_initial_solution(distance_matrix, x0)

    assert x == x0
    assert fx == expected_distance


@pytest.mark.parametrize(
    "distance_matrix", [distance_matrix1, distance_matrix2, distance_matrix3]
)
def test_setup_return_random_valid_solution(distance_matrix):
    """
    The setup outputs a random valid permutation if no initial solution
    is provided. This permutation must contain all nodes from 0 to n - 1
    and start at 0 (the root).
    """

    x, fx = setup_initial_solution(distance_matrix)

    assert set(x) == set(range(distance_matrix.shape[0]))
    assert x[0] == 0
    assert fx
