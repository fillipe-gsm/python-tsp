"""Contains typical distance matrices"""
from typing import List, Optional, Tuple

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

    Parameters
    ----------
    tsplib_file
        A string with the complete path of the TSPLIB file (or just its name if
        it is the in current path)

    Returns
    -------
    distance_matrix
        A ND-array with the equivalent distance matrix of the input file

    Raises
    ------
    NotImplementedError
        In case a not supported file is provided.

    Notes
    -----
    This function is able to handle the following types of TSPLIB files:
        - EDGE_WEIGHT_TYPE: "EUC_2D", "CEIL_", "GEO" and "EXPLICIT"
        - EDGE_WEIGHT_FORMAT: "FULL_MATRIX", "LOWER_ROW", "LOWER_DIAG_ROW",
        "UPPER_ROW" and "UPPER_DIAG_ROW".
    """
    with open(tsplib_file, "r") as f:
        lines = f.readlines()

    # Get file relevant attributes
    edge_weigth_type = _find_file_attribute(lines, "EDGE_WEIGHT_TYPE")
    edge_weigth_format = _find_file_attribute(lines, "EDGE_WEIGHT_FORMAT")

    if edge_weigth_type in ("EUC_2D", "CEIL_2D"):
        return _euc_2d_tsplib_distance_matrix(lines, edge_weigth_type)
    elif edge_weigth_type == "GEO":
        return _geo_tsplib_distance_matrix(lines)
    elif (
        edge_weigth_type == "EXPLICIT" and edge_weigth_format == "FULL_MATRIX"
    ):
        return _explicit_full_matrix_tsplib_distance_matrix(lines)
    elif (
        edge_weigth_type == "EXPLICIT"
        and edge_weigth_format in ("LOWER_ROW", "LOWER_DIAG_ROW")
    ):
        offset = 0 if edge_weigth_format == "LOWER_DIAG_ROW" else 1
        return _explicit_lower_row_tsplib_distance_matrix(lines, offset=offset)
    elif (
        edge_weigth_type == "EXPLICIT"
        and edge_weigth_format in ("UPPER_ROW", "UPPER_DIAG_ROW")
    ):
        offset = 0 if edge_weigth_format == "UPPER_DIAG_ROW" else 1
        return _explicit_upper_row_tsplib_distance_matrix(lines, offset=offset)
    else:
        raise NotImplementedError("tsplib file not supported")


def _find_file_attribute(lines: List[str], attribute: str) -> str:
    line = next((line for line in lines if line.startswith(attribute)), None)
    if line:
        return line.split(":")[1].strip()
    return ""


def _euc_2d_tsplib_distance_matrix(
    lines: List[str], edge_weigth_type: str
) -> np.ndarray:

    coordinates = _get_coordinates(lines)

    distance_matrix = euclidean_distance_matrix(coordinates)
    if edge_weigth_type == "CEIL_2D":
        distance_matrix = np.ceil(distance_matrix)

    return distance_matrix.astype(int)


def _geo_tsplib_distance_matrix(lines: List[str]) -> np.ndarray:

    coordinates = _get_coordinates(lines)
    distance_matrix = great_circle_distance_matrix(coordinates)

    return distance_matrix.astype(int)


def _get_coordinates(lines: List[str]) -> np.ndarray:
    """Get the coordinates in case of files without an explicit matrix"""
    dimension = int(_find_file_attribute(lines, "DIMENSION"))
    # Get the index of the line starting with the nodes information
    node_section_index = next(
        (
            i for i, line in enumerate(lines)
            if line.startswith("NODE_COORD_SECTION")
        )
    ) + 1

    def read_node_coordinates(line):
        return [float(coordinate) for coordinate in line.split()[1:]]

    return np.array([
        read_node_coordinates(line)
        for line in lines[node_section_index:node_section_index + dimension]
    ])


def _explicit_full_matrix_tsplib_distance_matrix(
    lines: List[str]
) -> np.ndarray:
    dimension = int(_find_file_attribute(lines, "DIMENSION"))
    distance_matrix_flattened = _get_distance_matrix_flattened(lines)

    distance_matrix = np.reshape(
        distance_matrix_flattened, (dimension, dimension)
    )
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix


def _explicit_lower_row_tsplib_distance_matrix(
    lines: List[str], offset: int = 0
) -> np.ndarray:
    dimension = int(_find_file_attribute(lines, "DIMENSION"))
    distance_matrix_flattened = _get_distance_matrix_flattened(lines)

    # Construct the lower part of the distance matrix
    distance_matrix = np.zeros((dimension, dimension))
    tril_indices = np.tril_indices(dimension, k=offset, m=dimension)
    distance_matrix[tril_indices] = distance_matrix_flattened
    distance_matrix += distance_matrix.transpose()
    return distance_matrix


def _explicit_upper_row_tsplib_distance_matrix(
    lines: List[str], offset: int = 0
) -> np.ndarray:
    dimension = int(_find_file_attribute(lines, "DIMENSION"))
    distance_matrix_flattened = _get_distance_matrix_flattened(lines)

    # Construct the upper part of the distance matrix
    distance_matrix = np.zeros((dimension, dimension))
    triu_indices = np.triu_indices(dimension, k=offset, m=dimension)
    distance_matrix[triu_indices] = distance_matrix_flattened
    distance_matrix += distance_matrix.transpose()
    return distance_matrix


def _get_distance_matrix_flattened(lines: List[str]) -> List[int]:
    """Return all cells of the distance matrix in the TSPLIB file"""

    # Get the index of the line starting with the edges information
    edge_section_index = next(
        (
            i for i, line in enumerate(lines)
            if line.startswith("EDGE_WEIGHT_SECTION")
        )
    ) + 1

    # Each file line does not necessarily contains one matrix row. Thus, read
    # all matrix data at once first and then reshape the array later
    def read_line_data(line):
        return [int(cell) for cell in line.split()]

    distance_matrix_flattened = []
    for line in lines[edge_section_index:]:
        # Try reading the current line and stop if a non-integer pops up (such
        # as "EOF" or another section)
        try:
            distance_matrix_flattened.extend(read_line_data(line))
        except ValueError:
            break

    return distance_matrix_flattened
