import pytest

from python_tsp.utils import setup_initial_solution
from tests.data import distance_matrix1


@pytest.mark.parametrize("starting_node", [0, 1, 2, 3, 4])
def test_initial_solution_respects_starting_node(starting_node):
    x0, _ = setup_initial_solution(
        distance_matrix1, starting_node=starting_node
    )

    # Ensure all nodes are contained in the solution and that
    # the starting node is as requested
    assert set(x0) == set(range(distance_matrix1.shape[0]))
    assert x0[0] == starting_node
