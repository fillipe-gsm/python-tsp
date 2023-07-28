import numpy as np
import pytest

from python_tsp.exact.branch_and_bound.node import Node

INF = np.iinfo(int).max


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
    invalid_matrix = np.full((5, 5), INF)
    response_matrix, response_cost = Node.compute_reduced_matrix(
        matrix=invalid_matrix
    )

    assert np.all(response_matrix == invalid_matrix)
    assert response_cost == 0


def test_create_node_from_cost_matrix(cost_matrix, reduced_cost_matrix):
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
    parent = Node.from_cost_matrix(cost_matrix=cost_matrix)
    response = Node.from_parent(parent=parent, index=index)

    assert response.level == 1
    assert response.index == index
    assert response.cost == expected_cost
    assert response.path == [parent.index, response.index]


@pytest.mark.parametrize("index, expected_cost", [(1, 35), (2, 53), (4, 31)])
def test_min_cost_node(cost_matrix, index, expected_cost):
    min_cost_node = Node.from_cost_matrix(cost_matrix=cost_matrix)
    response = Node.from_parent(parent=min_cost_node, index=index)

    assert min_cost_node.cost == 25
    assert response.cost == expected_cost
    assert min_cost_node < response
