import numpy as np
import pytest

from python_tsp.heuristics import local_search
from python_tsp.utils import compute_permutation_distance
from .data import (
    distance_matrix1, distance_matrix2, distance_matrix3,
    optimal_permutation1, optimal_permutation2, optimal_permutation3,
    optimal_distance1, optimal_distance2, optimal_distance3
)


perturbation_schemes = ["ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "two_opt"]


class TestImportModule:
    def test_can_import_local_search_solver(self):
        """
        Check if the main function can be imported from the `heuristics`
        package
        """
        from python_tsp.heuristics import (  # noqa
            solve_tsp_local_search
        )


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
        The setup outputs the same input if provided, together with its
        objective value
        """

        x0 = [0, 2, 3, 1, 4]
        x, fx = local_search.setup(distance_matrix, x0)

        assert x == x0
        assert fx == expected_distance

    @pytest.mark.parametrize(
        "distance_matrix",
        [distance_matrix1, distance_matrix2, distance_matrix3]
    )
    def test_setup_return_random_valid_solution(self, distance_matrix):
        """
        The setup outputs a random valid permutation if no initial solution
        is provided. This permutation must contain all nodes from 0 to n - 1
        and start at 0 (the root).
        """

        x, fx = local_search.setup(distance_matrix)

        assert set(x) == set(range(distance_matrix.shape[0]))
        assert x[0] == 0
        assert fx

    @pytest.mark.parametrize("scheme", perturbation_schemes)
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

        xopt, fopt = local_search.solve_tsp_local_search(
            distance_matrix, x, perturbation_scheme=scheme
        )

        assert fopt < fx

    @pytest.mark.parametrize("scheme", perturbation_schemes)
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
        xopt, fopt = local_search.solve_tsp_local_search(
            distance_matrix, x, perturbation_scheme=scheme
        )

        assert xopt == x
        assert fopt == fx

    @pytest.mark.parametrize("scheme", perturbation_schemes)
    def test_local_search_with_time_constraints(self, scheme, caplog):
        """
        The actual time execution tends to respect the provided limits, but
        it seems to vary a bit between platforms. For instance, locally it may
        take a few milisseconds more, but on Github it may be a few whole
        seconds.
        Thus, this test checks if a proper warning log is created if the time
        constraint stopped execution early.
        """

        max_processing_time = 1  # 1 second
        np.random.seed(1)  # for repeatability with the same distance matrix
        distance_matrix = np.random.rand(5000, 5000)  # very large matrix

        local_search.solve_tsp_local_search(
            distance_matrix,
            perturbation_scheme=scheme,
            max_processing_time=max_processing_time,
        )

        assert "Stopping early due to time constraints" in caplog.text

    def test_log_file_is_created_if_required(self, tmp_path):
        """
        If a log_file is provided, it contains information about the execution.
        """

        log_file = tmp_path / "tmp_log_file.log"

        local_search.solve_tsp_local_search(
            distance_matrix1, log_file=log_file
        )

        assert log_file.exists()
        assert "Current value" in log_file.read_text()
