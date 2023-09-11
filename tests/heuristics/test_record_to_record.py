import pytest

from python_tsp.heuristics import solve_tsp_record_to_record
from python_tsp.utils import compute_permutation_distance
from tests.data import (
    distance_matrix1,
    distance_matrix2,
    distance_matrix3,
    optimal_distance1,
    optimal_distance2,
    optimal_distance3,
    optimal_permutation1,
    optimal_permutation2,
    optimal_permutation3,
)


@pytest.mark.parametrize(
    "distance_matrix", [distance_matrix1, distance_matrix2, distance_matrix3]
)
def test_record_to_record_solution_is_valid(distance_matrix):
    """
    It is not possible to determine the returned solution, so this function
    just checks if it is valid: it has all nodes and begins at the root 0.
    """

    x, _ = solve_tsp_record_to_record(
        distance_matrix=distance_matrix,
    )

    assert set(x) == set(range(5))
    assert x[0] == 0


@pytest.mark.parametrize(
    "distance_matrix", [distance_matrix1, distance_matrix2, distance_matrix3]
)
def test_record_to_record_returns_better_neighbor(distance_matrix):
    """
    If there is room for improvement, a better neighbor is returned.
    Here, we choose purposely a permutation that can be improved.
    """
    x0 = [0, 4, 2, 3, 1]
    fx = compute_permutation_distance(
        distance_matrix=distance_matrix, permutation=x0
    )

    _, fopt = solve_tsp_record_to_record(
        distance_matrix=distance_matrix,
        x0=x0,
    )

    assert fopt < fx


@pytest.mark.parametrize(
    "distance_matrix, optimal_permutation, optimal_distance",
    [
        (distance_matrix1, optimal_permutation1, optimal_distance1),
        (distance_matrix2, optimal_permutation2, optimal_distance2),
        (distance_matrix3, optimal_permutation3, optimal_distance3),
    ],
)
def test_record_to_record_returns_equal_optimal_solution(
    distance_matrix, optimal_permutation, optimal_distance
):
    """
    If there is no room for improvement, the same solution is returned.
    Here, we choose purposely the optimal solution of each problem
    """
    xopt, fopt = solve_tsp_record_to_record(
        distance_matrix=distance_matrix,
        x0=optimal_permutation,
    )

    assert xopt == optimal_permutation
    assert fopt == optimal_distance


def test_record_to_record_log_file_is_created_if_required(tmp_path):
    """
    If a log_file is provided, it contains information about the execution.
    """

    log_file = tmp_path / "tmp_log_file.log"

    solve_tsp_record_to_record(
        distance_matrix=distance_matrix1,
        log_file=log_file,
        verbose=True,
    )

    assert log_file.exists()
    assert "Current value" in log_file.read_text()
