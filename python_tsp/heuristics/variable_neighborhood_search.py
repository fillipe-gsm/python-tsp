"""Simple variable neighborhood search solver"""
from typing import List, Optional, TextIO, Tuple

import numpy as np

from python_tsp.heuristics import solve_tsp_local_search
from python_tsp.heuristics.perturbation_schemes import neighborhood_gen
from python_tsp.utils import setup_initial_solution

PERTURBATION_SCHEMES = tuple(neighborhood_gen.keys())


def solve_tsp_variable_neighborhood_search(
    distance_matrix: np.ndarray,
    x0: Optional[List[int]] = None,
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[List[int], float]:
    """
    Solve the Traveling Salesman Problem (TSP) using
    the Variable Neighborhood Search (VNS) algorithm.

    Parameters
    ----------
    distance_matrix
        A 2D array representing the distance matrix between cities.
    x0
        An optional initial solution for the TSP. If not
        provided, a random initial solution will be generated.
    log_file
        The name of the log file to which the algorithm progress
        will be logged. If not provided, no log file will be generated.
    verbose
        If True, the algorithm's progress will be printed to the console.

    Returns
    -------
    Tuple
        A tuple containing the best found TSP solution (list of city indices)
        and its corresponding total distance (cost).

    References
    ----------
    [1] Éric D. Taillard.
    "Design of Heuristic Algorithms for Hard Optimization",
    Graduate Texts in Operations Research, Springer, 2023.
    """
    x, fx = setup_initial_solution(distance_matrix, x0)
    perturbation_index = 0
    log_file_handler = (
        open(log_file, "w", encoding="utf-8") if log_file else None
    )
    while perturbation_index < len(PERTURBATION_SCHEMES):
        perturbation_name = PERTURBATION_SCHEMES[perturbation_index]
        neighbors_gen = neighborhood_gen[perturbation_name](x)

        try:
            x_neighbor = next(neighbors_gen)
        except StopIteration:
            perturbation_index += 1
            continue

        msg = f"Current value: {fx}; Neighborhood scheme: {perturbation_name}"
        _print_message(msg, verbose, log_file_handler)

        xn, fn = solve_tsp_local_search(
            distance_matrix=distance_matrix,
            x0=x_neighbor,
            perturbation_scheme=perturbation_name,
            verbose=verbose,
        )

        if fn < fx:
            x, fx = xn, fn
            perturbation_index = 0
            msg = "Improvement found!"
        else:
            perturbation_index += 1
            msg = (
                "No improvements found! Checking another neighborhood scheme."
            )
        _print_message(msg, verbose, log_file_handler)

    if log_file_handler:
        log_file_handler.close()

    return x, fx


def _print_message(
    msg: str, verbose: bool, log_file_handler: Optional[TextIO]
) -> None:
    """
    Print a message to the console and/or log it to a file if provided.

    Parameters
    ----------
    msg
        The message to be printed/logged.
    verbose
        If True, the message will be printed to the console.
    log_file_handler
        An optional file handler for the log file. If provided, the
        message will be logged to the file.
    """
    if log_file_handler:
        print(msg, file=log_file_handler)

    if verbose:
        print(msg)
