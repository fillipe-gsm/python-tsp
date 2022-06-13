import numpy as np
import pytest

from python_tsp.distances import euclidean_distance_matrix


@pytest.fixture
def sources():
    return np.array([[1, -1], [2, -2], [3, -3], [4, -4]])


@pytest.fixture
def destinations():
    return np.array([[5, -5], [6, -6], [7, -7]])


def test_distance_is_euclidean():
    """It must return an actual Euclidean distance
    In this case, it is easy to see that the distance from [1, 1] to [4, 5]
    is:
        sqrt((4 - 1)**2 + (5 - 1)**2) = sqrt(3**2 + 4**2) = 5
    """
    source = np.array([1, 1])
    destination = np.array([4, 5])

    distance_matrix = euclidean_distance_matrix(source, destination)

    assert distance_matrix[0] == 5.


def test_all_elements_are_non_negative(sources, destinations):
    """Being distances, all elements must be non-negative"""
    distance_matrix = euclidean_distance_matrix(sources, destinations)

    assert np.all(distance_matrix >= 0)


def test_square_matrix_has_zero_diagonal(sources):
    """Main diagonal is the distance from a point to itself"""
    distance_matrix = euclidean_distance_matrix(sources)

    assert np.all(np.diag(distance_matrix) == 0)


def test_square_matrix_is_symmetric(sources):
    distance_matrix = euclidean_distance_matrix(sources)

    assert np.allclose(distance_matrix, distance_matrix.T)


def test_matrix_has_proper_shape(sources, destinations):
    """N sources and M destinations should produce an (N x M) array"""
    distance_matrix = euclidean_distance_matrix(sources, destinations)

    N, M = sources.shape[0], destinations.shape[0]
    assert distance_matrix.shape == (N, M)
