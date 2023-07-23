import numpy as np
import pytest

from python_tsp.exact.branch_and_bound import Node, compute_reduced_matrix

INF = np.iinfo(int).max


@pytest.fixture
def matrix():
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
    Test the 'compute_reduced_matrix' function with different matrices.

    This test verifies the correctness of the 'compute_reduced_matrix'
    function for both valid and reduced matrices.

    Test steps:
    1. For each (request_matrix, expected_reduced_matrix, expected_cost)
    tuple in the test cases:
        a. Call the 'compute_reduced_matrix' function with the
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
        # Step 1a: Call the 'compute_reduced_matrix'
        # function with the 'request_matrix'.
        response_matrix, response_cost = compute_reduced_matrix(
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
    Test the 'compute_reduced_matrix' function with invalid matrices.

    The function should handle invalid matrices correctly by not reducing
    any rows or columns, and the total reduction cost should be zero.

    Test steps:
    1. Create an invalid matrix filled with INF values.
    2. Pass the invalid matrix to the 'compute_reduced_matrix' function.
    3. Check if the resulting 'response_matrix' is equal to the input
    'invalid_matrix', indicating that no reduction was performed.
    4. Check if the 'response_cost' is zero, as no reduction was done.

    """
    # Step 1: Create an invalid matrix filled with INF values.
    invalid_matrix = np.full((5, 5), INF)

    # Step 2: Pass the invalid matrix to the 'compute_reduced_matrix' function.
    response_matrix, response_cost = compute_reduced_matrix(
        matrix=invalid_matrix
    )

    # Step 3: Check if the resulting 'response_matrix' is
    # equal to the input 'invalid_matrix'.
    assert np.all(response_matrix == invalid_matrix)

    # Step 4: Check if the 'response_cost' is zero, as no reduction was done.
    assert response_cost == 0


def test_create_root_node(matrix, reduced_matrix):
    response_node = Node(parent_node=None, vertex=0)
    response_node.reduced_matrix, response_node.cost = compute_reduced_matrix(
        matrix=matrix
    )

    assert np.all(response_node.reduced_matrix == reduced_matrix)
    assert response_node.cost == 25
    assert response_node.level == 0
    assert response_node.vertex == 0
    assert response_node.path == [0]


@pytest.mark.parametrize("vertex,expected_cost", [(1, 35), (2, 53), (4, 31)])
def test_create_child_node(matrix, vertex, expected_cost):
    root = Node(parent_node=None, vertex=0)
    root.reduced_matrix, root.cost = compute_reduced_matrix(matrix=matrix)
    response_child_node = Node(parent_node=root, vertex=vertex)

    assert root < response_child_node
    assert response_child_node.cost == expected_cost
    assert response_child_node.level == 1
    assert response_child_node.vertex == vertex
    assert response_child_node.path == [root.vertex, vertex]
