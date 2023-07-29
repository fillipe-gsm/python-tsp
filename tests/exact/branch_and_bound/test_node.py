import numpy as np
import pytest

from python_tsp.exact.branch_and_bound import Node
from math import inf


@pytest.fixture
def cost_matrix():
    """A cost matrix"""
    return np.array(
        [
            [inf, 20, 30, 10, 11],
            [15, inf, 16, 4, 2],
            [3, 5, inf, 2, 4],
            [19, 6, 18, inf, 3],
            [16, 4, 7, 16, inf],
        ]
    )


@pytest.fixture
def reduced_cost_matrix():
    """Reduced matrix corresponding to the cost matrix"""
    return np.array(
        [
            [inf, 10, 17, 0, 1],
            [12, inf, 11, 2, 0],
            [0, 3, inf, 0, 2],
            [15, 3, 12, inf, 0],
            [11, 0, 0, 12, inf],
        ]
    )


def test_compute_reduced_matrix(cost_matrix, reduced_cost_matrix):
    """
    Test the `compute_reduced_matrix` function of the `Node` class.

    Check if the function correctly calculates the reduced cost matrix and
    the total reduction cost when provided with an original cost matrix.

    Test cases:
        1. The original matrix should be reduced with a cost of 25.
        2. An already reduced matrix should remain unchanged with a
           cost of 0.
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
    Test the `compute_reduced_matrix` function of the `Node` class with
    invalid matrices.

    Check if the function returns the same invalid matrix and a reduction
    cost of 0 when provided with an invalid matrix (filled with infinite
    values).
    """
    invalid_matrix = np.full((5, 5), inf)
    response_matrix, response_cost = Node.compute_reduced_matrix(
        matrix=invalid_matrix
    )

    assert np.all(response_matrix == invalid_matrix)
    assert response_cost == 0


def test_create_node_from_cost_matrix(cost_matrix, reduced_cost_matrix):
    """
    Test the `from_cost_matrix` function of the `Node` class.

    Check if the function creates a new node correctly from an original
    cost matrix.

    Verifications:
        - The new node should have the level (level) equal to 0.
        - The new node should have the index (index) equal to 0.
        - The new node should have the cost (cost) equal to 25.
        - The new node should have the correct reduced cost matrix.
        - The new node should have the path (path) [0].
    """
    response = Node.from_cost_matrix(cost_matrix=cost_matrix)

    assert response.level == 0
    assert response.index == 0
    assert response.cost == 25
    assert np.all(response.cost_matrix == reduced_cost_matrix)
    assert response.path == [0]


@pytest.mark.parametrize(
    "index, expected_cost", [(1, 35), (2, 53), (3, 25), (4, 31)]
)
def test_create_node_from_parent(cost_matrix, index, expected_cost):
    """
    Test the `from_parent` function of the `Node` class.

    Check if the function creates a new node (child) correctly from an
    existing parent node.

    Verifications:
        - The new node should have the level (level) equal to 1.
        - The new node should have the index (index) equal to the
          provided value.
        - The new node should have the cost (cost) equal to the
          expected cost.
        - The new node should have the correct path, including the parent
          node index.
    """
    parent = Node.from_cost_matrix(cost_matrix=cost_matrix)
    response = Node.from_parent(parent=parent, index=index)

    assert response.level == 1
    assert response.index == index
    assert response.cost == expected_cost
    assert response.path == [parent.index, response.index]


@pytest.mark.parametrize("index, expected_cost", [(1, 35), (2, 53), (4, 31)])
def test_min_cost_node(cost_matrix, index, expected_cost):
    """
    Test the comparison operator `<` between nodes.

    Check if the node with the lowest cost is correctly identified
    among two nodes.

    Verifications:
        - The initial parent node should have a cost equal to 25.
        - The new node created from the initial parent node should have the
          expected cost.
        - The initial parent node should be considered smaller than the
          new node.
    """
    min_cost_node = Node.from_cost_matrix(cost_matrix=cost_matrix)
    response = Node.from_parent(parent=min_cost_node, index=index)

    assert min_cost_node.cost == 25
    assert response.cost == expected_cost
    assert min_cost_node < response
