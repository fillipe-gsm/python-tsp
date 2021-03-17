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
    euc_2d_file = "tests/tsplib_data/a280.tsp"
    ceil_2d_file = "tests/tsplib_data/dsj1000ceil.tsp"
    geo_file = "tests/tsplib_data/gr666.tsp"
    explicit_full_matrix_file = "tests/tsplib_data/br17.atsp"
    explicit_lower_diag_row_file = "tests/tsplib_data/gr48.tsp"
    explicit_upper_row_file = "tests/tsplib_data/brazil58.tsp"
    explicit_upper_diag_row_file = "tests/tsplib_data/si1032.tsp"

    def test_euc_2d_tsplib_file(self):
        """The symmetric test problem corresponds to a280, with 280 nodes"""
        dimension = 280
        distance_matrix = distances.tsplib_distance_matrix(self.euc_2d_file)

        assert distance_matrix.shape == (dimension, dimension)
        assert distance_matrix.dtype == int

    def test_ceil_2d_tsplib_file(self):
        """
        The symmetric test problem corresponds to dsf1000ceil, with 1000 nodes
        """
        dimension = 1000
        distance_matrix = distances.tsplib_distance_matrix(self.ceil_2d_file)

        assert distance_matrix.shape == (dimension, dimension)
        assert distance_matrix.dtype == int

    def test_geo_tsplib_file(self):
        """
        """
        dimension = 666
        distance_matrix = distances.tsplib_distance_matrix(self.geo_file)

        assert distance_matrix.shape == (dimension, dimension)
        assert distance_matrix.dtype == int

    def test_explicit_full_matrix_tsplib_file(self):
        """The asymmetric test problem corresponds to br17, with 17 nodes """
        dimension = 17
        distance_matrix = distances.tsplib_distance_matrix(
            self.explicit_full_matrix_file
        )

        assert distance_matrix.shape == (dimension, dimension)
        assert np.array_equal(distance_matrix.diagonal(), np.zeros(dimension))

    def test_explicit_lower_diag_row_tsplib_file(self):
        """The symmetric test problem corresponds to gr48, with 48 nodes"""
        dimension = 48
        distance_matrix = distances.tsplib_distance_matrix(
            self.explicit_lower_diag_row_file
        )

        assert distance_matrix.shape == (dimension, dimension)
        assert np.array_equal(distance_matrix.diagonal(), np.zeros(dimension))

    def test_explicit_upper_row_tsplib_file(self):
        """The symmetric test problem corresponds to brazil58, with 58 nodes"""
        dimension = 58
        distance_matrix = distances.tsplib_distance_matrix(
            self.explicit_upper_row_file
        )

        assert distance_matrix.shape == (dimension, dimension)
        assert np.array_equal(distance_matrix.diagonal(), np.zeros(dimension))

    def test_explicit_upper_diag_row_tsplib_file(self):
        """The symmetric test problem corresponds to si1032, with 1032 nodes"""
        dimension = 1032
        distance_matrix = distances.tsplib_distance_matrix(
            self.explicit_upper_diag_row_file
        )

        assert distance_matrix.shape == (dimension, dimension)
