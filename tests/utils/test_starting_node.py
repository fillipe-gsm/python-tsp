import pytest

from python_tsp.utils import setup_initial_solution
from python_tsp.utils.setup_initial_solution import STARTING_NODE_TOO_LARGE_MSG
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


def test_exception_is_raise_if_starting_node_is_too_large():
    with pytest.raises(ValueError) as e:
        x0, _ = setup_initial_solution(
            distance_matrix1, starting_node=999
        )

    assert str(e.value) == STARTING_NODE_TOO_LARGE_MSG
