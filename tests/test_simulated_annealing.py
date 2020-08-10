import pytest

from python_tsp.heuristics import simulated_annealing
from python_tsp.utils import compute_permutation_distance
from .data import (
    distance_matrix1, distance_matrix2, distance_matrix3,
)


class TestSimulatedAnnealing:
    x = [0, 1, 2, 3, 4]

    @pytest.mark.parametrize("scheme", ["ps1", "ps2", "ps3"])
    def test_perturbation_generates_appropriate_neighbor(self, scheme):
        """
        Check if the generated neighbor has the same nodes as the input in any
        order
        """

        xn = simulated_annealing._perturbation(
            self.x, perturbation_scheme=scheme
        )

        assert set(xn) == set(self.x)

    @pytest.mark.parametrize(
        "distance_matrix",
        [distance_matrix1, distance_matrix2, distance_matrix3]
    )
    @pytest.mark.parametrize("scheme", ["ps1", "ps2", "ps3"])
    def test_initial_temperature(self, distance_matrix, scheme):
        """
        The initial temperature is mostly random, so simply check if it is
        valid (greater than zero).
        """

        fx = compute_permutation_distance(distance_matrix, self.x)
        temp = simulated_annealing.initial_temperature(
            distance_matrix, self.x, fx, perturbation_scheme=scheme
        )

        assert temp > 0

    def test_acceptance_rule_improvement(self):
        """If fn < fx, the acceptance rule must return True"""

        flag = simulated_annealing.acceptance_rule(2, 1, 1)
        assert flag

    @pytest.mark.parametrize(
        "distance_matrix",
        [distance_matrix1, distance_matrix2, distance_matrix3]
    )
    @pytest.mark.parametrize("scheme", ["ps1", "ps2", "ps3"])
    def test_simulated_annealing_solution(self, distance_matrix, scheme):
        """
        It is not possible to determine the returned solution, so this function
        just checks if it is valid and has all nodes
        """

        x, _ = simulated_annealing.solve_tsp_simulated_annealing(
            distance_matrix, perturbation_scheme=scheme
        )

        assert set(x) == set(range(5))
