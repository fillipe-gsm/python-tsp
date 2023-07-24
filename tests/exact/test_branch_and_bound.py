import numpy as np
import pytest

from python_tsp.exact.branch_and_bound import Node, NodePriorityQueue

INF = np.iinfo(int).max


@pytest.fixture
def matrix():
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
def reduced_matrix():
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


def test_compute_reduced_matrix(matrix, reduced_matrix):
    """
    Test the 'Node.compute_reduced_matrix' function with different matrices.

    This test verifies the correctness of the 'Node.compute_reduced_matrix'
    function for both valid and reduced matrices.

    Test steps:
    1. For each (request_matrix, expected_reduced_matrix, expected_cost)
    tuple in the test cases:
        a. Call the 'Node.compute_reduced_matrix' function with the
        'request_matrix'.
        b. Check if the resulting 'response_matrix' is equal to the
        'expected_reduced_matrix', verifying that the function performs the
        reduction correctly.
        c. Check if the 'response_cost' is equal to the 'expected_cost',
        ensuring that the function correctly computes the total reduction cost.
    """
    for request_matrix, expected_reduced_matrix, expected_cost in [
        (
            matrix,
            reduced_matrix,
            25,
        ),  # Original matrix should be reduced with a cost of 25.
        (
            reduced_matrix,
            reduced_matrix,
            0,
        ),  # Already reduced matrix should remain unchanged with a cost of 0.
    ]:
        # Step 1a: Call the 'Node.compute_reduced_matrix'
        # function with the 'request_matrix'.
        response_matrix, response_cost = Node.compute_reduced_matrix(
            matrix=request_matrix
        )

        # Step 1b: Check if the resulting 'response_matrix' is
        # equal to the 'expected_reduced_matrix'.
        assert np.all(response_matrix == expected_reduced_matrix)

        # Step 1c: Check if the 'response_cost' is equal to the
        # 'expected_cost'.
        assert response_cost == expected_cost


def test_compute_reduced_matrix_with_invalid_matrices():
    """
    Test the 'Node.compute_reduced_matrix' function with invalid matrices.

    The function should handle invalid matrices correctly by not reducing
    any rows or columns, and the total reduction cost should be zero.

    Test steps:
    1. Create an invalid matrix filled with INF values.
    2. Pass the invalid matrix to the 'Node.compute_reduced_matrix' function.
    3. Check if the resulting 'response_matrix' is equal to the input
    'invalid_matrix', indicating that no reduction was performed.
    4. Check if the 'response_cost' is zero, as no reduction was done.

    """
    # Step 1: Create an invalid matrix filled with INF values.
    invalid_matrix = np.full((5, 5), INF)

    # Step 2: Pass the invalid matrix to the
    # 'Node.compute_reduced_matrix' function.
    response_matrix, response_cost = Node.compute_reduced_matrix(
        matrix=invalid_matrix
    )

    # Step 3: Check if the resulting 'response_matrix' is
    # equal to the input 'invalid_matrix'.
    assert np.all(response_matrix == invalid_matrix)

    # Step 4: Check if the 'response_cost' is zero, as no reduction was done.
    assert response_cost == 0


def test_create_node_from_cost_matrix(matrix, reduced_matrix):
    """
    Checks if the `Node.root_from_cost_matrix` method correctly creates a
    new Node as the root node for a graph traversal, based on the provided
    cost matrix. It verifies if the attributes of the created node match the
    expected values.
    """
    response_node = Node.root_from_cost_matrix(cost_matrix=matrix)

    assert response_node.level == 0
    assert response_node.index == 0
    assert response_node.cost == 25
    assert np.all(response_node.reduced_matrix == reduced_matrix)
    assert response_node.path == [0]


@pytest.mark.parametrize(
    "node_index, node_expected_cost", [(1, 35), (2, 53), (3, 25), (4, 31)]
)
def test_create_child_node(matrix, node_index, node_expected_cost):
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
    root = Node.root_from_cost_matrix(cost_matrix=matrix)
    response_node = Node.from_parent_node(
        parent_node=root, node_index=node_index
    )

    assert response_node.level == 1
    assert response_node.index == node_index
    assert response_node.cost == node_expected_cost
    assert response_node.path == [root.index, response_node.index]


def test_nodes_priority_queue(matrix):
    """
    Checks the functionality of the NodePriorityQueue by creating a priority
    queue and adding child nodes to it based on the root node. It then
    verifies if the queue is not empty and if the first node popped from the
    queue has the expected cost value.

    The test ensures that the priority queue correctly maintains the order
    of Node objects based on their priority (cost) and pops the node with
    the highest priority (lowest cost) first.
    """
    pq = NodePriorityQueue()
    node_indices = range(len(matrix))
    root = Node.root_from_cost_matrix(cost_matrix=matrix)

    for node_index in node_indices:
        if root.reduced_matrix[root.index][node_index] != INF:
            child_node = Node.from_parent_node(
                parent_node=root, node_index=node_index
            )
            pq.push(child_node)

    assert not pq.empty
    assert pq.pop().cost == 25
