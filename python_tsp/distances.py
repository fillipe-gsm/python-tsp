"""Compute distance matrices for given sets of points"""

import numpy as np


def euclidean_distance_matrix(
    sources: np.ndarray, destinations: np.ndarray
) -> np.ndarray:
    """Euclidean distance matrix

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
