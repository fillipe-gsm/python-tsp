"""Module with a brute force TSP solver"""
from itertools import permutations
from typing import Any, List, Optional, Tuple

import numpy as np

from python_tsp.utils import compute_permutation_distance


def solve_tsp_brute_force(
    distance_matrix: np.ndarray, starting_node: int = 0,
) -> Tuple[Optional[List], Any]:
    """Solve TSP to optimality with a brute force approach

    Parameters
    ----------
    distance_matrix
        Distance matrix of shape (n x n) with the (i, j) entry indicating the
        distance from node i to j. It does not need to be symmetric
    
    starting_node
        Determines the starting node of the final permutation. Defaults to 0.

    Returns
    -------
    A permutation of nodes from 0 to n that produces the least total
    distance

    The total distance the optimal permutation produces

    Notes
    ----
    The algorithm checks all permutations and returns the one with smallest
    distance. In principle, the total number of possibilities would be n! for
    n nodes. However, we can fix node 0 and permutate only the remaining,
    reducing the possibilities to (n - 1)!.
    """

    # Exclude `starting_node` from the range since it is fixed
    other_points = [
        node for node in range(distance_matrix.shape[0])
        if node != starting_node
    ]
    best_distance = np.inf
    best_permutation = None

    for partial_permutation in permutations(other_points):
        # Remember to add the starting node before evaluating it
        permutation = [starting_node] + list(partial_permutation)
        distance = compute_permutation_distance(distance_matrix, permutation)

        if distance < best_distance:
            best_distance = distance
            best_permutation = permutation

    return best_permutation, best_distance
