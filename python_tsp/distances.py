"""Contains typical distance matrices"""

import numpy as np

EARTH_RADIUS_M = 6371000  # Earth radius in meters


def euclidean_distance_matrix(
    sources: np.ndarray, destinations: np.ndarray
) -> np.ndarray:
    """Distance matrix using the Euclidean distance

    Parameters
    ----------
    sources, destinations
        Arrays with each row containing the coordinates of a point

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
    # Ensure at least two dimensions for the following vectorized code work
    sources = np.atleast_2d(sources)
    destinations = np.atleast_2d(destinations)

    return np.sqrt(
        ((sources[:, :, None] - destinations[:, :, None].T) ** 2).sum(1)
    )


def great_circle_distance_matrix(
    sources: np.ndarray, destinations: np.ndarray
) -> np.ndarray:
    """Distance matrix using the Great Circle distance
    This is an Euclidean-like distance but on spheres [1]. In this case it is
    used to estimate the distance in meters between locations in the Earth.

    Parameters
    ----------
    sources, destinations
        Arrays with each row containing the coordinates of a point in the form
        [lat, lng]. Notice it only considers the first two columns.

    Returns
    -------
    distance_matrix
        Array with the (i. j) entry indicating the Great Circle distance (in
        meters) between the i-th row in `sources` and the j-th row in
        `destinations`.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Great-circle_distance
    Using the second computational formula
    """
    # Ensure at least two dimensions for the following vectorized code work
    sources = np.atleast_2d(sources)
    destinations = np.atleast_2d(destinations)

    sources_rad = np.radians(sources)
    dests_rad = np.radians(destinations)

    # Define variables for better readability
    delta_phi = sources_rad[:, [0]] - dests_rad[:, 0]  # (N x M) lat difference
    delta_lambda = sources_rad[:, [1]] - dests_rad[:, 1]  # (N x M) lng
    phi1 = sources_rad[:, [0]]  # (N x 1) array of source latitudes
    phi2 = dests_rad[:, 0]  # (1 x M) array of destination latitudes

    delta_sigma = 2*np.arcsin(np.sqrt(
        np.sin(delta_phi/2)**2
        + np.cos(phi1)*np.cos(phi2)*np.sin(delta_lambda/2)**2
    ))
    return EARTH_RADIUS_M*delta_sigma
