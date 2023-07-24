import numpy as np
import pytest

from python_tsp.exact import solve_tsp_branch_and_bound
from python_tsp.exact.branch_and_bound import Node, NodePriorityQueue
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

INF = np.iinfo(int).max


@pytest.fixture
def distance_matrix():
    """A distance matrix"""
    return np.array(
        [
            [0, 20, 30, 10, 11],
            [15, 0, 16, 4, 2],
            [3, 5, 0, 2, 4],
            [19, 6, 18, 0, 3],
            [16, 4, 7, 16, 0],
        ]
    )


@pytest.fixture
def cost_matrix():
    """A cost matrix"""
    return np.array(
        [
            [INF, 20, 30, 10, 11],
            [15, INF, 16, 4, 2],
            [3, 5, INF, 2, 4],
            [19, 6, 18, INF, 3],
            [16, 4, 7, 16, INF],
        ]
    )


@pytest.fixture
def reduced_cost_matrix():
    """Reduced matrix corresponding to the cost matrix"""
    return np.array(
        [
            [INF, 10, 17, 0, 1],
            [12, INF, 11, 2, 0],
            [0, 3, INF, 0, 2],
            [15, 3, 12, INF, 0],
            [11, 0, 0, 12, INF],
        ]
    )


def test_compute_reduced_matrix(cost_matrix, reduced_cost_matrix):
    """
    Test the 'Node.compute_reduced_matrix' function with different matrices.

    This test verifies the correctness of the 'Node.compute_reduced_matrix'
    function for both valid and reduced matrices.
    """
    for request_matrix, expected_reduced_matrix, expected_cost in [
        (
            cost_matrix,
            reduced_cost_matrix,
            25,
        ),  # Original matrix should be reduced with a cost of 25.
        (
            reduced_cost_matrix,
            reduced_cost_matrix,
            0,
        ),  # Already reduced matrix should remain unchanged with a cost of 0.
    ]:
        response_matrix, response_cost = Node.compute_reduced_matrix(
            matrix=request_matrix
        )

        assert np.all(response_matrix == expected_reduced_matrix)
        assert response_cost == expected_cost


def test_compute_reduced_matrix_with_invalid_matrices():
    """
    Test the 'Node.compute_reduced_matrix' function with invalid matrices.

    The function should handle invalid matrices correctly by not reducing
    any rows or columns, and the total reduction cost should be zero.
    """
    invalid_matrix = np.full((5, 5), INF)
    response_matrix, response_cost = Node.compute_reduced_matrix(
        matrix=invalid_matrix
    )

    assert np.all(response_matrix == invalid_matrix)
    assert response_cost == 0


def test_create_node_from_cost_matrix(cost_matrix, reduced_cost_matrix):
    """
    Checks if the `Node.root_from_cost_matrix` method correctly creates a
    new Node as the root node for a graph traversal, based on the provided
    cost matrix. It verifies if the attributes of the created node match the
    expected values.
    """
    response_node = Node.root_from_cost_matrix(cost_matrix=cost_matrix)

    assert response_node.level == 0
    assert response_node.index == 0
    assert response_node.cost == 25
    assert np.all(response_node.reduced_matrix == reduced_cost_matrix)
    assert response_node.path == [0]


@pytest.mark.parametrize(
    "node_index, node_expected_cost", [(1, 35), (2, 53), (3, 25), (4, 31)]
)
def test_create_child_node(cost_matrix, node_index, node_expected_cost):
    """
    Checks if the `Node.from_parent_node` method correctly creates a
    new child Node based on a parent Node and a specified index. It
    verifies if the attributes of the created child Node match the expected
    values.

    The test creates a root Node using the provided cost matrix, and then
    generates a new child Node based on the root Node and the specified
    node_index. The child Node's attributes are compared with the
    expected values.
    """
    root = Node.root_from_cost_matrix(cost_matrix=cost_matrix)
    response_node = Node.from_parent_node(
        parent_node=root, node_index=node_index
    )

    assert response_node.level == 1
    assert response_node.index == node_index
    assert response_node.cost == node_expected_cost
    assert response_node.path == [root.index, response_node.index]


def test_nodes_priority_queue(cost_matrix):
    """
    Checks the functionality of the NodePriorityQueue by creating a priority
    queue and adding child nodes to it based on the root node. It then
    verifies if the queue is not empty and if the first node popped from the
    queue has the expected cost value.

    The test ensures that the priority queue correctly maintains the order
    of Node objects based on their priority (cost) and pops the node with
    the highest priority (lowest cost) first.
    """
    node_indices = range(len(cost_matrix))
    pq = NodePriorityQueue()
    root = Node.root_from_cost_matrix(cost_matrix=cost_matrix)

    for node_index in node_indices:
        is_live_node = root.reduced_matrix[root.index][node_index] != INF
        if is_live_node:
            live_node = Node.from_parent_node(
                parent_node=root, node_index=node_index
            )
            pq.push(live_node)

    assert not pq.empty
    assert pq.pop().cost == 25


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
    "distance_matrix, expected_permutation, expected_distance",
    [
        (distance_matrix1, optimal_permutation1, optimal_distance1),
        (distance_matrix2, optimal_permutation2, optimal_distance2),
        (distance_matrix3, optimal_permutation3, optimal_distance3),
    ],
)
def test_solution_is_optimal(
    distance_matrix, expected_permutation, expected_distance
):
    """This exact method should return an optimal solution"""
    permutation, _ = solve_tsp_branch_and_bound(
        distance_matrix=distance_matrix
    )

    assert (
        compute_permutation_distance(distance_matrix, permutation)
        == expected_distance
    )
