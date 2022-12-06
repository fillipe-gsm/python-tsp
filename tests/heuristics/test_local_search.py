import sys
from io import StringIO

import numpy as np
import pytest

from python_tsp.heuristics import local_search
from python_tsp.heuristics.perturbation_schemes import neighborhood_gen
from python_tsp.utils import compute_permutation_distance
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


PERTURBATION_SCHEMES = neighborhood_gen.keys()


@pytest.mark.parametrize("scheme", PERTURBATION_SCHEMES)
@pytest.mark.parametrize(
    "distance_matrix", [distance_matrix1, distance_matrix2, distance_matrix3]
)
def test_local_search_returns_better_neighbor(scheme, distance_matrix):
    """
    If there is room for improvement, a better neighbor is returned.
    Here, we choose purposely a permutation that can be improved.
    """
    x = [0, 4, 2, 3, 1]
    fx = compute_permutation_distance(distance_matrix, x)

    _, fopt = local_search.solve_tsp_local_search(
        distance_matrix, x, perturbation_scheme=scheme
    )

    assert fopt < fx


@pytest.mark.parametrize("scheme", PERTURBATION_SCHEMES)
@pytest.mark.parametrize(
    "distance_matrix, optimal_permutation, optimal_distance",
    [
        (distance_matrix1, optimal_permutation1, optimal_distance1),
        (distance_matrix2, optimal_permutation2, optimal_distance2),
        (distance_matrix3, optimal_permutation3, optimal_distance3),
    ],
)
def test_local_search_returns_equal_optimal_solution(
    scheme, distance_matrix, optimal_permutation, optimal_distance
):
    """
    If there is no room for improvement, the same solution is returned.
    Here, we choose purposely the optimal solution of each problem
    """
    x = optimal_permutation
    fx = optimal_distance
    xopt, fopt = local_search.solve_tsp_local_search(
        distance_matrix, x, perturbation_scheme=scheme
    )

    assert xopt == x
    assert fopt == fx


@pytest.mark.parametrize("scheme", PERTURBATION_SCHEMES)
def test_local_search_with_time_constraints(scheme):
    """
    The actual time execution tends to respect the provided limits, but
    it seems to vary a bit between platforms. For instance, locally it may
    take a few milisseconds more, but on Github it may be a few whole
    seconds.
    Thus, this test checks if a proper warning is printed if the time
    constraint stopped execution early.
    """

    max_processing_time = 1  # 1 second
    np.random.seed(1)  # for repeatability with the same distance matrix
    distance_matrix = np.random.rand(5000, 5000)  # very large matrix

    captured_output = StringIO()  # Create StringIO object
    sys.stdout = captured_output  # and redirect stdout.

    local_search.solve_tsp_local_search(
        distance_matrix,
        perturbation_scheme=scheme,
        max_processing_time=max_processing_time,
        verbose=True,
    )

    assert local_search.TIME_LIMIT_MSG in captured_output.getvalue()


def test_log_file_is_created_if_required(tmp_path):
    """
    If a log_file is provided, it contains information about the execution.
    """

    log_file = tmp_path / "tmp_log_file.log"

    local_search.solve_tsp_local_search(distance_matrix1, log_file=log_file)

    assert log_file.exists()
    assert "Current value" in log_file.read_text()
