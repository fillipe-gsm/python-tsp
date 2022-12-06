from typing import Optional

import numpy as np

from .data_processing import process_input


EARTH_RADIUS_METERS = 6371000


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

    sources, destinations = process_input(sources, destinations)
    sources_rad = np.radians(sources)
    dests_rad = np.radians(destinations)

    delta_lambda = sources_rad[:, [1]] - dests_rad[:, 1]  # (N x M) lng
    phi1 = sources_rad[:, [0]]  # (N x 1) array of source latitudes
    phi2 = dests_rad[:, 0]  # (1 x M) array of destination latitudes

    delta_sigma = np.arctan2(
        np.sqrt(
            (np.cos(phi2) * np.sin(delta_lambda)) ** 2
            + (
                np.cos(phi1) * np.sin(phi2)
                - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)
            )
            ** 2
        ),
        (
            np.sin(phi1) * np.sin(phi2)
            + np.cos(phi1) * np.cos(phi2) * np.cos(delta_lambda)
        ),
    )

    return EARTH_RADIUS_METERS * delta_sigma
