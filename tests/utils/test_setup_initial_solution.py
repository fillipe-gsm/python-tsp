import pytest

import numpy as np

from python_tsp.utils import setup_initial_solution
from python_tsp.utils.setup_initial_solution import (
    STARTING_NODE_OUTSIDE_BOUNDARIES_MSG,
    ENDING_NODE_OUTSIDE_BOUNDARIES_MSG,
    STARTING_ENDING_NODE_ARE_EQUAL_MSG,
    INVALID_SIZE_INITIAL_SOLUTION_MSG,
    INVALID_INITIAL_SOLUTION_MSG,
    OVERLAPPING_INPUT_ARGUMENTS_MSG,
)
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


def test_setup__respects_starting_node():
    num_nodes = 10
    starting_node = 5
    distance_matrix = np.random.rand(num_nodes, num_nodes)

    x0, fx0 = setup_initial_solution(
        distance_matrix, starting_node=starting_node
    )

    assert x0[0] == starting_node
    assert fx0


def test_setup_input_validation__starting_node_outside_boundaries():
    num_nodes = 10
    distance_matrix = np.random.rand(num_nodes, num_nodes)

    with pytest.raises(ValueError) as exc:
        setup_initial_solution(distance_matrix, starting_node=-1)
    assert STARTING_NODE_OUTSIDE_BOUNDARIES_MSG in str(exc)

    with pytest.raises(ValueError) as exc:
        setup_initial_solution(distance_matrix, starting_node=num_nodes)
    assert STARTING_NODE_OUTSIDE_BOUNDARIES_MSG in str(exc)


def test_setup_input_validation__ending_node_outside_boundaries():
    num_nodes = 10
    distance_matrix = np.random.rand(num_nodes, num_nodes)

    with pytest.raises(ValueError) as exc:
        setup_initial_solution(distance_matrix, ending_node=-1)
    assert ENDING_NODE_OUTSIDE_BOUNDARIES_MSG in str(exc)

    with pytest.raises(ValueError) as exc:
        setup_initial_solution(distance_matrix, ending_node=num_nodes)
    assert ENDING_NODE_OUTSIDE_BOUNDARIES_MSG in str(exc)


def test_setup_input_validation__starting_ending_nodes_equal():
    num_nodes = 10
    distance_matrix = np.random.rand(num_nodes, num_nodes)

    with pytest.raises(ValueError) as exc:
        setup_initial_solution(distance_matrix, starting_node=5, ending_node=5)
    assert STARTING_ENDING_NODE_ARE_EQUAL_MSG in str(exc)


def test_setup_input_validation__x0_has_wrong_size():
    num_nodes = 10
    distance_matrix = np.random.rand(num_nodes, num_nodes)

    with pytest.raises(ValueError) as exc:
        setup_initial_solution(distance_matrix, x0=list(range(num_nodes + 1)))
    assert INVALID_SIZE_INITIAL_SOLUTION_MSG in str(exc)

    with pytest.raises(ValueError) as exc:
        setup_initial_solution(distance_matrix, x0=list(range(num_nodes - 1)))
    assert INVALID_SIZE_INITIAL_SOLUTION_MSG in str(exc)


def test_setup_input_validation__x0_does_not_contain_all_nodes():
    num_nodes = 10
    distance_matrix = np.random.rand(num_nodes, num_nodes)
    x0 = list(range(num_nodes))
    x0[5] = num_nodes + 5  # replace with a node outside 0:num_nodes-1

    with pytest.raises(ValueError) as exc:
        setup_initial_solution(distance_matrix, x0=x0, starting_node=0)
    assert INVALID_INITIAL_SOLUTION_MSG in str(exc)


def test_setup_input_validation__overlapping_arguments():
    num_nodes = 10
    distance_matrix = np.random.rand(num_nodes, num_nodes)
    x0 = list(range(num_nodes))

    with pytest.raises(ValueError) as exc:
        setup_initial_solution(distance_matrix, x0=x0, starting_node=0)
    assert OVERLAPPING_INPUT_ARGUMENTS_MSG in str(exc)

    with pytest.raises(ValueError) as exc:
        setup_initial_solution(distance_matrix, x0=x0, ending_node=0)
    assert OVERLAPPING_INPUT_ARGUMENTS_MSG in str(exc)

    with pytest.raises(ValueError) as exc:
        setup_initial_solution(distance_matrix, x0=x0, starting_node=5, ending_node=0)
    assert OVERLAPPING_INPUT_ARGUMENTS_MSG in str(exc)
