import numpy as np

from python_tsp.distances import tsplib_distance_matrix


EUC_2D_FILE = "tests/tsplib_data/a280.tsp"
CEIL_2D_FILE = "tests/tsplib_data/dsj1000ceil.tsp"
GEO_FILE = "tests/tsplib_data/ulysses22.tsp"
EXPLICIT_FULL_MATRIX_FILE = "tests/tsplib_data/br17.atsp"
EXPLICIT_LOWER_DIAG_ROW_FILE = "tests/tsplib_data/gr48.tsp"
EXPLICIT_UPPER_ROW_FILE = "tests/tsplib_data/brazil58.tsp"
EXPLICIT_UPPER_DIAG_ROW_FILE = "tests/tsplib_data/si1032.tsp"


def test_euc_2d_tsplib_file():
    dimension = 280
    distance_matrix = tsplib_distance_matrix(EUC_2D_FILE)

    assert distance_matrix.shape == (dimension, dimension)
    assert distance_matrix.dtype == int


def test_ceil_2d_tsplib_file():
    dimension = 1000
    distance_matrix = tsplib_distance_matrix(CEIL_2D_FILE)

    assert distance_matrix.shape == (dimension, dimension)
    assert distance_matrix.dtype == int


def test_geo_tsplib_file():
    dimension = 22
    distance_matrix = tsplib_distance_matrix(GEO_FILE)

    assert distance_matrix.shape == (dimension, dimension)
    assert distance_matrix.dtype == int


def test_explicit_full_matrix_tsplib_file():
    """Note that this test file is asymetric"""
    dimension = 17
    distance_matrix = tsplib_distance_matrix(EXPLICIT_FULL_MATRIX_FILE)

    assert distance_matrix.shape == (dimension, dimension)
    assert np.array_equal(distance_matrix.diagonal(), np.zeros(dimension))
    assert not np.array_equal(distance_matrix, distance_matrix.T)


def test_explicit_lower_diag_row_tsplib_file():
    dimension = 48
    distance_matrix = tsplib_distance_matrix(EXPLICIT_LOWER_DIAG_ROW_FILE)

    assert distance_matrix.shape == (dimension, dimension)
    assert np.array_equal(distance_matrix.diagonal(), np.zeros(dimension))


def test_explicit_upper_row_tsplib_file():
    dimension = 58
    distance_matrix = tsplib_distance_matrix(EXPLICIT_UPPER_ROW_FILE)

    assert distance_matrix.shape == (dimension, dimension)
    assert np.array_equal(distance_matrix.diagonal(), np.zeros(dimension))


def test_explicit_upper_diag_row_tsplib_file():
    dimension = 1032
    distance_matrix = tsplib_distance_matrix(EXPLICIT_UPPER_DIAG_ROW_FILE)

    assert distance_matrix.shape == (dimension, dimension)
    assert np.array_equal(distance_matrix.diagonal(), np.zeros(dimension))
