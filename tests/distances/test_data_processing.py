import numpy as np

from python_tsp.distances.data_processing import process_input


def test_1d_array_becomes_2d():
    source = np.array([1, -1])
    destination = np.array([5, -5])

    sources_out, destinations_out = process_input(
        source, destination
    )

    assert sources_out.shape == (1, 2)
    assert destinations_out.shape == (1, 2)


def test_no_destinations_become_sources():

    sources = np.array([[1, -1], [2, -2], [3, -3], [4, -4]])

    sources_out, destinations_out = process_input(sources)

    assert np.array_equal(sources_out, sources)
    assert np.array_equal(destinations_out, sources)
