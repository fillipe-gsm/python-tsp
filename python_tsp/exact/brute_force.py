"""Module with a brute force TSP solver"""

from itertools import permutations
from typing import List, Tuple

import numpy as np

from python_tsp.utils import compute_permutation_distance


def solve_tsp_brute_force(distance_matrix: np.ndarray) -> Tuple[List, float]:
    """Solve TSP to optimality with a brute force approach

    Parameters
    ----------
    distance_matrix
        Distance matrix of shape (n x n) with the (i, j) entry indicating the
        distance from node i to j. It does not need to be symmetric

    Returns
    -------
    permutation
        A permutation of nodes from 0 to n that produces the least total
        distance

    distance
        The total distance the optimal permutation produces

    Notes
    ----
    The algorithm checks all permutations and returns the one with smallest
    distance. In principle, the total number of possibilities would be n! for
    n nodes. However, we can fix node 0 and permutate only the remaining,
    reducing the possibilities to (n - 1)!.
    """

    # Get all points from 1 to n (fix 0 as mentioned in the notes)
    points = range(1, distance_matrix.shape[0])
    best_distance = np.inf
    best_solution = None
    for partial_permutation in permutations(points):
        # Add 0 node to the partial permutation and convert to a list
        permutation = [0] + list(partial_permutation)
        distance = compute_permutation_distance(permutation, distance_matrix)
        if distance < best_distance:
            best_distance = distance
            best_solution = permutation

    return best_solution, best_distance
