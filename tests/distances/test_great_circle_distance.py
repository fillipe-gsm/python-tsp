import numpy as np
import pytest

from python_tsp.distances import great_circle_distance_matrix


@pytest.fixture
def sources():
    return np.array([[1, -1], [2, -2], [3, -3], [4, -4]])


@pytest.fixture
def destinations():
    return np.array([[5, -5], [6, -6], [7, -7]])


def test_all_elements_are_non_negative(sources, destinations):
    """Being distances, all elements must be non-negative"""
    distance_matrix = great_circle_distance_matrix(sources, destinations)

    assert np.all(distance_matrix >= 0)


def test_square_matrix_has_zero_diagonal(sources):
    """Main diagonal is the distance from a point to itself"""
    distance_matrix = great_circle_distance_matrix(sources)

    assert np.all(np.diag(distance_matrix) == 0)


def test_square_matrix_is_symmetric(sources):
    distance_matrix = great_circle_distance_matrix(sources, sources)

    assert np.allclose(distance_matrix, distance_matrix.T)


def test_matrix_has_proper_shape(sources, destinations):
    """N sources and M destinations should produce an (N x M) array"""
    distance_matrix = great_circle_distance_matrix(sources, destinations)

    N, M = sources.shape[0], destinations.shape[0]
    assert distance_matrix.shape == (N, M)


def test_distance_works_with_1d_arrays(sources, destinations):
    """The code is vectorized for 2d arrays, but should work for 1d as well
    """
    source = sources[0]
    destination = destinations[0]

    great_circle_distance_matrix(source, destination)
