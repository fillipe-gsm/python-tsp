"""Contains typical distance matrices"""
from typing import Optional, Tuple

import numpy as np
import tsplib95


EARTH_RADIUS_METERS = 6371000


def euclidean_distance_matrix(
    sources: np.ndarray, destinations: Optional[np.ndarray] = None
) -> np.ndarray:
    """Distance matrix using the Euclidean distance

    Parameters
    ----------
    sources, destinations
        Arrays with each row containing the coordinates of a point. If
        ``destinations`` is None, compute the distance between each source in
        ``sources`` and outputs a square distance matrix.

    Returns
    -------
    distance_matrix
        Array with the (i, j) entry indicating the Euclidean distance between
        the i-th row in ``sources`` and the j-th row in ``destinations``.

    Notes
    -----
    The Euclidean distance between points x = (x1, x2, ..., xn) and
    y = (y1, y2, ..., yn), with n coordinates each, is given by:

        sqrt((y1 - x1)**2 + (y2 - x2)**2 + ... + (yn - xn)**2)

    If the user requires the distance between each point in a single array,
    call this this function with ``destinations`` set to `None`.
    """
    sources, destinations = _process_input(sources, destinations)
    return np.sqrt(
        ((sources[:, :, None] - destinations[:, :, None].T) ** 2).sum(axis=1)
    )


def great_circle_distance_matrix(
    sources: np.ndarray, destinations: Optional[np.ndarray] = None
) -> np.ndarray:
    """Distance matrix using the Great Circle distance
    This is an Euclidean-like distance but on spheres [1]. In this case it is
    used to estimate the distance in meters between locations in the Earth.

    Parameters
    ----------
    sources, destinations
        Arrays with each row containing the coordinates of a point in the form
        [lat, lng]. Notice it only considers the first two columns.
        Also, if ``destinations`` is `None`, compute the distance between each
        source in ``sources``.

    Returns
    -------
    distance_matrix
        Array with the (i, j) entry indicating the Great Circle distance (in
        meters) between the i-th row in ``sources`` and the j-th row in
        ``destinations``.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Great-circle_distance
    Using the third computational formula
    """

    sources, destinations = _process_input(sources, destinations)
    sources_rad = np.radians(sources)
    dests_rad = np.radians(destinations)

    delta_lambda = sources_rad[:, [1]] - dests_rad[:, 1]  # (N x M) lng
    phi1 = sources_rad[:, [0]]  # (N x 1) array of source latitudes
    phi2 = dests_rad[:, 0]  # (1 x M) array of destination latitudes

    delta_sigma = np.arctan2(
        np.sqrt(
            (np.cos(phi2) * np.sin(delta_lambda))**2 +
            (np.cos(phi1) * np.sin(phi2) -
             np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda))**2
        ),
        (np.sin(phi1) * np.sin(phi2) +
         np.cos(phi1) * np.cos(phi2) * np.cos(delta_lambda))
    )

    return EARTH_RADIUS_METERS * delta_sigma


def _process_input(
    sources: np.ndarray, destinations: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-process input
    This function ensures ``sources`` and ``destinations`` have at least two
    dimensions, and if ``destinations`` is `None`, set it equal to ``sources``.
    """
    if destinations is None:
        destinations = sources

    sources = np.atleast_2d(sources)
    destinations = np.atleast_2d(destinations)

    return sources, destinations


def tsplib_distance_matrix(tsplib_file: str) -> np.ndarray:
    """Distance matrix from a TSPLIB file

    Parameters
    ----------
    tsplib_file
        A string with the complete path of the TSPLIB file (or just its name if
        it is the in current path)

    Returns
    -------
    distance_matrix
        A ND-array with the equivalent distance matrix of the input file

    Notes
    -----
    This function can handle any file supported by the `tsplib95` lib.
    """

    tsp_problem = tsplib95.load(tsplib_file)
    distance_matrix_flattened = np.array([
        tsp_problem.get_weight(*edge) for edge in tsp_problem.get_edges()
    ])
    distance_matrix = np.reshape(
        distance_matrix_flattened,
        (tsp_problem.dimension, tsp_problem.dimension),
    )

    # Some problems with EXPLICIT matrix have a large number in the distance
    # from a node to itself, which makes no sense for our problems. Thus,
    # always ensure a diagonal filled with zeros
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix
