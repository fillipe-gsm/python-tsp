"""Auxiliar functions that may be used in most modules"""
from typing import List

import numpy as np


def compute_permutation_distance(
    distance_matrix: np.ndarray, permutation: List[int]
) -> float:
    """Compute the total route distance of a given permutation

    Parameters
    ----------
    distance_matrix
        Distance matrix of shape (n x n) with the (i, j) entry indicating the
        distance from node i to j. It does not need to be symmetric

    permutation
        A list with nodes from 0 to n - 1 in any order

    Returns
    -------
    Total distance of the path given in ``permutation`` for the provided
    ``distance_matrix``

    Notes
    -----
    Suppose the permutation [0, 1, 2, 3], with four nodes. The total distance
    of this path will be from 0 to 1, 1 to 2, 2 to 3, and 3 back to 0. This
    can be fetched from a distance matrix using:

        distance_matrix[ind1, ind2], where
        ind1 = [0, 1, 2, 3]  # the FROM nodes
        ind2 = [1, 2, 3, 0]  # the TO nodes

    This can easily be generalized to any permutation by using ind1 as the
    given permutation, and moving the first node to the end to generate ind2.
    """
    ind1 = permutation
    ind2 = permutation[1:] + permutation[:1]
    return distance_matrix[ind1, ind2].sum()
