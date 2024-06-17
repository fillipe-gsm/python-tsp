"""Euclidean distance"""

from typing import Optional

import numpy as np

from .data_processing import process_input


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
    sources, destinations = process_input(sources, destinations)
    return np.sqrt(
        ((sources[:, :, None] - destinations[:, :, None].T) ** 2).sum(axis=1)
    )
