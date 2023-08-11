import pytest

from python_tsp.heuristics import solve_tsp_variable_neighborhood_search
from tests.data import distance_matrix1, distance_matrix2, distance_matrix3


@pytest.mark.parametrize(
    "distance_matrix", [distance_matrix1, distance_matrix2, distance_matrix3]
)
def test_variable_neighborhood_search_solution_is_valid(distance_matrix):
    """
    It is not possible to determine the returned solution, so this function
    just checks if it is valid: it has all nodes and begins at the root 0.
    """

    x, _ = solve_tsp_variable_neighborhood_search(
        distance_matrix, verbose=True
    )

    assert set(x) == set(range(5))
    assert x[0] == 0


def test_log_file_is_created_if_required(tmp_path):
    """
    If a log_file is provided, it contains information about the execution.
    """

    log_file = tmp_path / "tmp_log_file.log"

    solve_tsp_variable_neighborhood_search(
        distance_matrix1, log_file=log_file, verbose=True
    )

    assert log_file.exists()
    assert "Current value:" in log_file.read_text()
    assert "Search neighborhood:" in log_file.read_text()
