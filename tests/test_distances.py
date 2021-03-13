import numpy as np

from python_tsp import distances


# Arrays used in the tests
sources = np.array([[1, -1], [2, -2], [3, -3], [4, -4]])
destinations = np.array([[5, -5], [6, -6], [7, -7]])


class TestInputPreprocessing:
    def test_1d_array_becomes_2d(self):
        source = sources[0]
        destination = destinations[0]

        sources_out, destinations_out = distances._process_input(
            source, destination
        )

        assert sources_out.shape == (1, 2)
        assert destinations_out.shape == (1, 2)

    def test_no_destinations_become_sources(self):

        sources_out, destinations_out = distances._process_input(sources)

        assert np.array_equal(sources_out, sources)
        assert np.array_equal(destinations_out, sources)


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

    def test_all_elements_are_non_negative(self):
        """Being distances, all elements must be non-negative"""
        distance_matrix = distances.euclidean_distance_matrix(
            sources, destinations
        )

        assert np.all(distance_matrix >= 0)

    def test_square_matrix_has_zero_diagonal(self):
        """Main diagonal is the distance from a point to itself"""
        distance_matrix = distances.euclidean_distance_matrix(sources)

        assert np.all(np.diag(distance_matrix) == 0)

    def test_square_matrix_is_symmetric(self):
        distance_matrix = distances.euclidean_distance_matrix(sources)

        assert np.allclose(distance_matrix, distance_matrix.T)

    def test_matrix_has_proper_shape(self):
        """N sources and M destinations should produce an (N x M) array"""
        distance_matrix = distances.euclidean_distance_matrix(
            sources, destinations
        )

        N, M = sources.shape[0], destinations.shape[0]
        assert distance_matrix.shape == (N, M)


class TestGreatCircleDistanceMatrix:
    def test_all_elements_are_non_negative(self):
        """Being distances, all elements must be non-negative"""
        distance_matrix = distances.great_circle_distance_matrix(
            sources, destinations
        )

        assert np.all(distance_matrix >= 0)

    def test_square_matrix_has_zero_diagonal(self):
        """Main diagonal is the distance from a point to itself"""
        distance_matrix = distances.great_circle_distance_matrix(sources)

        assert np.all(np.diag(distance_matrix) == 0)

    def test_square_matrix_is_symmetric(self):
        distance_matrix = distances.great_circle_distance_matrix(
            sources, sources
        )

        assert np.allclose(distance_matrix, distance_matrix.T)

    def test_matrix_has_proper_shape(self):
        """N sources and M destinations should produce an (N x M) array"""
        distance_matrix = distances.great_circle_distance_matrix(
            sources, destinations
        )

        N, M = sources.shape[0], destinations.shape[0]
        assert distance_matrix.shape == (N, M)

    def test_distance_works_with_1d_arrays(self):
        """The code is vectorized for 2d arrays, but should work for 1d as well
        """
        source = sources[0]
        destination = destinations[0]

        distances.great_circle_distance_matrix(source, destination)


class TestTSPLIBDistanceMatrix:
    symmetric_tsplib_file = "tests/tsplib_data/symmetric_test_file.tsp"
    asymmetric_tsplib_file = "tests/tsplib_data/asymmetric_test_file.atsp"

    def test_symmetric_tsplib_matrix_conversion(self):
        """
        The symmetric test problem corresponds to a280, with 280 nodes.
        Check its dimensions and whether it contains only integer values (as
        specified by the TSPLIB documentation).
        """
        distance_matrix = distances.tsplib_distance_matrix(
            self.symmetric_tsplib_file
        )

        assert distance_matrix.shape == (280, 280)
        assert distance_matrix.dtype == int

    def test_asymmetric_tsplib_matrix_conversion(self):
        """
        The asymmetric test problem corresponds to br17, with 17 nodes.
        Check its dimensions and set the main diagonal to 0 as opposed to a
        large number as in most instances.
        """
        distance_matrix = distances.tsplib_distance_matrix(
            self.asymmetric_tsplib_file
        )

        assert distance_matrix.shape == (17, 17)
