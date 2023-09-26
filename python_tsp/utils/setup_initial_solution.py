from random import sample
from typing import List, Optional, Tuple

import numpy as np

from .permutation_distance import compute_permutation_distance


def setup_initial_solution(
    distance_matrix: np.ndarray,
    x0: Optional[List] = None,
    starting_node: int = 0
) -> Tuple[List[int], float]:
    """Return initial solution and its objective value

    Parameters
    ----------
    distance_matrix
        Distance matrix of shape (n x n) with the (i, j) entry indicating the
        distance from node i to j

    x0
        Permutation of nodes from 0 to n - 1 indicating the starting solution.
        If not provided, a random list is created.

    Returns
    -------
    x0
        Permutation with initial solution. If ``x0`` was provided, it is the
        same list

    fx0
        Objective value of x0
    """

    if not x0:
        n = distance_matrix.shape[0]  # number of nodes
        x0 = _build_initial_permutation(n, starting_node)

    fx0 = compute_permutation_distance(distance_matrix, x0)
    return x0, fx0


def _build_initial_permutation(n: int, starting_node: int) -> List[int]:
    """
    Build a random list of integers from 0 to `n` - 1 guaranteeing the initial
    node is `starting_node`.
    """
    all_nodes_except_starting_node = [
        node for node in range(n) if node != starting_node
    ]
    x0 = [starting_node] + sample(all_nodes_except_starting_node, n - 1)

    return x0
