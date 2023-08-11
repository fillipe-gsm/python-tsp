"""Simple variable neighborhood search solver"""
from typing import List, Optional, TextIO, Tuple

import numpy as np

from python_tsp.heuristics import solve_tsp_local_search
from python_tsp.heuristics.perturbation_schemes import neighborhood_gen
from python_tsp.utils import setup_initial_solution

# All available neighborhood schemes for the local search algorithm
AVAILABLE_PERTURBATION_SCHEMES = sorted(neighborhood_gen.keys(), reverse=True)


def solve_tsp_variable_neighborhood_search(
    distance_matrix: np.ndarray,
    x0: Optional[List[int]] = None,
    perturbation_schemes: Optional[List[str]] = None,
    max_processing_time: Optional[float] = None,
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

    perturbation_schemes
        An optional list of perturbation schemes to be used in the VNS
        algorithm. The listed neighborhood schemes must be available to
        the local search algorithm, the order of the list matters.
        If not provided, all available schemes will be used.

    max_processing_time
        An optional maximum processing time in seconds. If not provided, the
        method stops only when a local minimum is obtained.

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
    log_file_handler = (
        open(log_file, "w", encoding="utf-8") if log_file else None
    )

    x, fx = setup_initial_solution(distance_matrix, x0)
    perturbation_index = 0
    perturbation_schemes = (
        perturbation_schemes or AVAILABLE_PERTURBATION_SCHEMES
    )
    while perturbation_index < len(perturbation_schemes):
        perturbation_name = perturbation_schemes[perturbation_index]

        msg = f"Current value: {fx}; Search neighborhood: {perturbation_name}"
        _print_message(msg, verbose, log_file_handler)

        x_neighbor = next(neighborhood_gen[perturbation_name](x), None)
        if x_neighbor:
            xn, fn = solve_tsp_local_search(
                distance_matrix=distance_matrix,
                x0=x_neighbor,
                max_processing_time=max_processing_time,
            )
            if fn < fx:
                x, fx = xn, fn
                perturbation_index = 0
                continue

        perturbation_index += 1

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
