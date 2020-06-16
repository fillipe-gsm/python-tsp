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

    def test_square_matrix_has_zero_diagonal(self, sources):
        """Main diagonal is the distance from a point to itself"""
        distance_matrix = distances.euclidean_distance_matrix(sources, sources)

        assert np.all(np.diag(distance_matrix) == 0)

    def test_square_matrix_is_symmetric(self, sources):
        distance_matrix = distances.euclidean_distance_matrix(sources, sources)

        assert np.allclose(distance_matrix, distance_matrix.T)

    def test_matrix_has_proper_shape(self, sources, destinations):
        """N sources and M destinations should produce an (N x M) array"""
        distance_matrix = distances.euclidean_distance_matrix(
            sources, destinations
        )

        N, M = sources.shape[0], destinations.shape[0]
        assert distance_matrix.shape == (N, M)


class TestGreatCircleDistanceMatrix:
    def test_all_elements_are_non_negative(self, sources, destinations):
        """Being distances, all elements must be non-negative"""
        distance_matrix = distances.great_circle_distance_matrix(
            sources, destinations
        )

        assert np.all(distance_matrix >= 0)

    def test_square_matrix_has_zero_diagonal(self, sources):
        """Main diagonal is the distance from a point to itself"""
        distance_matrix = distances.great_circle_distance_matrix(
            sources, sources
        )

        assert np.all(np.diag(distance_matrix) == 0)

    def test_square_matrix_is_symmetric(self, sources):
        distance_matrix = distances.great_circle_distance_matrix(
            sources, sources
        )

        assert np.allclose(distance_matrix, distance_matrix.T)

    def test_matrix_has_proper_shape(self, sources, destinations):
        """N sources and M destinations should produce an (N x M) array"""
        distance_matrix = distances.great_circle_distance_matrix(
            sources, destinations
        )

        N, M = sources.shape[0], destinations.shape[0]
        assert distance_matrix.shape == (N, M)

    def test_distance_works_with_1d_arrays(self, sources, destinations):
        """The code is vectorized for 2d arrays, but should work for 1d as well
        """
        source = sources[0]
        destination = destinations[0]

        distances.great_circle_distance_matrix(source, destination)
