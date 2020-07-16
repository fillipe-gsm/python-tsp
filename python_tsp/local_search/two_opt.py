"""Module with a 2-opt local search solver"""
from typing import List, Tuple, Optional
from random import shuffle

import numpy as np

from python_tsp.utils import compute_permutation_distance


def solve_tsp_two_opt(
    distance_matrix: np.ndarray, initial_permutation: Optional[List] = None
) -> Tuple[List, float]:
    """Solve a TSP problem with a local search using 2-opt

    Notes
    -----
    Here are the steps of the algorithm:
        1. Let x, fx be the initial permutation and its distance;
        2. Perform a neighborhood search in x:
            2.1 For each x' neighbor of x, if fx' < fx, set x <- x' and stop;
        3. Repeat step 2 until all neighbors of x are tried and there is no
        improvement. Return x, fx as solution.
    """

    x, fx = _initial_solution(distance_matrix, initial_permutation)

    flag = True
    while flag:
        flag, x, fx = _neighborhood_search(distance_matrix, x, fx)

    # Return x and fx as local optima
    return x, fx


def _initial_solution(
    distance_matrix: np.ndarray, initial_permutation: Optional[List] = None
) -> Tuple[List[int], np.ndarray]:
    """Return initial solution and its distance value
    If not provided, output a random permutation
    """

    if not initial_permutation:
        initial_permutation = list(range(distance_matrix.shape[0]))
        shuffle(initial_permutation)

    distance = compute_permutation_distance(
        distance_matrix, initial_permutation
    )

    return initial_permutation, distance


def _neighborhood_search(
    distance_matrix: np.ndarray, x: List[int], fx: float
) -> Tuple[bool, List[int], float]:
    """Perform neighborhood search and return the best neighbor

    Parameters
    ----------
    distance_matrix

    x
        Current best permutation

    fx
        Distance value of `x`

    Returns
    -------
    flag
        True if a better neighbor was found, False otherwise

    xn
        A possibly improved neighbor of `x`, or `x` itself otherwise

    fn
        Distance of `xn`
    """

    n = len(x)  # number of nodes
    for i in range(n - 1):
        for j in range(i, n):
            xn = _two_opt_swap(x, i, j)
            fn = compute_permutation_distance(distance_matrix, xn)
            if fn < fx:
                return True, xn, fn

    # If we got here, no improvement was made
    return False, x, fx


def _two_opt_swap(permutation: List[int], i: int, j: int) -> List[int]:
    """2-opt swap in a permutation

    Parameters
    ----------
    permutation
        A list with a permutation of nodes from 0 to n

    i, j
        Indices within n to perform the swap

    Returns
    -------
    A perturbed input permutation

    Notes
    -----
    Given a permutation of nodes 0 to n and indices i and j, this perturbation
    goes like:
        1. take permutation[0] to permutation[i-1] in current order
        2. take permutation[i] to permutation[j-1] in reverse order
        3. take permutation[j] to permutation[n] in current order
    """

    if i > j:
        raise ValueError(f"i cannot be greater than j ({i} > {j})")

    return permutation[:i] + list(reversed(permutation[i:j])) + permutation[j:]
