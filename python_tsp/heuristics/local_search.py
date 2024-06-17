"""Simple local search solver"""

from timeit import default_timer
from typing import List, Optional, Tuple, TextIO

import numpy as np

from python_tsp.utils import (
    compute_permutation_distance,
    setup_initial_solution,
)
from python_tsp.heuristics.perturbation_schemes import neighborhood_gen


TIME_LIMIT_MSG = "WARNING: Stopping early due to time constraints"


def solve_tsp_local_search(
    distance_matrix: np.ndarray,
    x0: Optional[List[int]] = None,
    perturbation_scheme: str = "two_opt",
    max_processing_time: Optional[float] = None,
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[List, float]:
    """Solve a TSP problem with a local search heuristic

    Parameters
    ----------
    distance_matrix
        Distance matrix of shape (n x n) with the (i, j) entry indicating the
        distance from node i to j

    x0
        Initial permutation. If not provided, it starts with a random path

    perturbation_scheme {"ps1", "ps2", "ps3", "ps4", "ps5", "ps6", ["two_opt"]}
        Mechanism used to generate new solutions. Defaults to "two_opt"

    max_processing_time {None}
        Maximum processing time in seconds. If not provided, the method stops
        only when a local minimum is obtained

    log_file
        If not `None`, creates a log file with details about the whole
        execution

    verbose
        If true, prints algorithm status every iteration

    Returns
    -------
    A permutation of nodes from 0 to n - 1 that produces the least total
    distance obtained (not necessarily optimal).

    The total distance the returned permutation produces.

    Notes
    -----
    Here are the steps of the algorithm:
        1. Let `x`, `fx` be a initial solution permutation and its objective
        value;
        2. Perform a neighborhood search in `x`:
            2.1 For each `x'` neighbor of `x`, if `fx'` < `fx`, set `x` <- `x'`
            and stop;
        3. Repeat step 2 until all neighbors of `x` are tried and there is no
        improvement. Return `x`, `fx` as solution.
    """
    x, fx = setup_initial_solution(distance_matrix, x0)
    max_processing_time = max_processing_time or np.inf

    log_file_handler = (
        open(log_file, "w", encoding="utf-8") if log_file else None
    )

    tic = default_timer()
    stop_early = False
    improvement = True

    while improvement and (not stop_early):
        improvement = False
        for n_index, xn in enumerate(neighborhood_gen[perturbation_scheme](x)):
            if default_timer() - tic > max_processing_time:
                _print_message(TIME_LIMIT_MSG, verbose, log_file_handler)
                stop_early = True
                break

            fn = compute_permutation_distance(distance_matrix, xn)

            msg = f"Current value: {fx}; Neighbor: {n_index}"
            _print_message(msg, verbose, log_file_handler)

            if fn < fx:
                improvement = True
                x, fx = xn, fn
                break  # early stop due to first improvement local search

    if log_file_handler:
        log_file_handler.close()

    return x, fx


def _print_message(
    msg: str, verbose: bool, log_file_handler: Optional[TextIO]
) -> None:
    if log_file_handler:
        print(msg, file=log_file_handler)

    if verbose:
        print(msg)
