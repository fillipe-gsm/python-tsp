import numpy as np
import pytest

from python_tsp import distances


@pytest.fixture
def sources():
    """An array to be used as sources in the tests"""
    return np.array([[1, -1], [2, -2], [3, -3], [4, -4]])


@pytest.fixture
def destinations():
    """An array to be used as destinations in the tests"""
    return np.array([[5, -5], [6, -6], [7, -7]])


class TestEuclideanDistanceMatrix:
    def test_distance_is_euclidean(self):
        """It must return an actual Euclidean distance
        In this case, it is easy to see that the distance from [1, 1] to [4, 5]
        is::
            sqrt((4 - 1)**2 + (5 - 1)**2) = sqrt(3**2 + 4**2) = 5
        """
        sources = np.array([1, 1])
        destinations = np.array([4, 5])

        distance_matrix = distances.euclidean_distance_matrix(
            sources, destinations
        )

        assert distance_matrix[0] == 5.

    def test_all_elements_are_non_negative(self, sources, destinations):
        """Being distances, all elements must be non-negative"""
        distance_matrix = distances.euclidean_distance_matrix(
            sources, destinations
        )

        assert np.all(distance_matrix >= 0)

    def test_diagonal_has_zeros_with_square_matrix(self, sources):
        """Main diagonal is the distance from a point to itself"""
        distance_matrix = distances.euclidean_distance_matrix(sources, sources)

        assert np.all(np.diag(distance_matrix) == 0)

    def test_matrix_has_proper_shape(self, sources, destinations):
        """N sources and M destinations should produce an (N x M) array"""
        distance_matrix = distances.euclidean_distance_matrix(
            sources, destinations
        )

        N, M = sources.shape[0], destinations.shape[0]
        assert distance_matrix.shape == (N, M)
