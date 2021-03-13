"""Contains typical distance matrices"""
from typing import Optional, TextIO, Tuple

import numpy as np

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
        ``sources``.

    Returns
    -------
    distance_matrix
        Array with the (i. j) entry indicating the Euclidean distance between
        the i-th row in `sources` and the j-th row in `destinations`.

    Notes
    -----
    The Euclidean distance between points x = (x1, x2, ..., xn) and
    y = (y1, y2, ..., yn), with n coordinates each, is given by:

        sqrt((y1 - x1)**2 + (y2 - x2)**2 + ... + (yn - xn)**2)

    If the user requires the distance between each point in a single array,
    call this this function with `sources` = `destinations`.
    """
    sources, destinations = _process_input(sources, destinations)
    return np.sqrt(
        ((sources[:, :, None] - destinations[:, :, None].T) ** 2).sum(1)
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
        Array with the (i. j) entry indicating the Great Circle distance (in
        meters) between the i-th row in `sources` and the j-th row in
        `destinations`.

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
    Currently, this function can handle files with types "TSP" and "ATSP".

    Parameters
    ----------
    tsplib_file
        A string with the complete path of the TSPLIB file (or just its name if
        it is the in current path)

    Returns
    -------
    distance_matrix
        A ND-array with the equivalent distance matrix of the input file
    """
    with open(tsplib_file, "r") as f:
        # Determine the type of the file
        for line in f:
            if line.startswith("TYPE"):
                _, tsp_type = line.split(":")
                break

        if tsp_type.strip() == "ATSP":  # strip() to remove \n and spaces
            return _asymmetric_tsplib_distance_matrix(f, tsplib_file)
        return _symmetric_tsplib_distance_matrix(f, tsplib_file)


def _symmetric_tsplib_distance_matrix(
    f: TextIO, tsplib_file: str
) -> np.ndarray:
    """Handles TSPLIB files of the type TSP (symmetric instances)"""
    # Discard lines until we get to the coordinates section
    for line in f:
        if line.startswith("NODE_COORD_SECTION"):
            break

    def read_node_coordinates(line):
        _, xstr, ystr = line.split()
        return (int(xstr), int(ystr))

    coordinates = np.array([
        read_node_coordinates(line) for line in f if not line.startswith("EOF")
    ])

    return euclidean_distance_matrix(coordinates).astype(int)


def _asymmetric_tsplib_distance_matrix(
    f: TextIO, tsplib_file: str
) -> np.ndarray:
    """Handles TSPLIB files of the type ATSP (asymmetric instances)"""
    # Discard lines until we get to the edges section
    for line in f:
        if line.startswith("DIMENSION"):
            _, nstr = line.split(":")
            n = int(nstr)
        if line.startswith("EDGE_WEIGHT_SECTION"):
            break

    def read_cells_line(line):
        return np.array([int(cell) for cell in line.split()])

    # Read each line in f and get the matrix cells until all elements of a row
    # are gathered (i.e., the row has `n` elements)
    row = []
    rows = []
    for line in f:
        if line.startswith("EOF"):
            break

        row.extend(read_cells_line(line))
        if len(row) == n:
            rows.append(row)
            row = []

    distance_matrix = np.array(rows)
    np.fill_diagonal(distance_matrix, 0)

    return distance_matrix
