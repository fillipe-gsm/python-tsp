from random import randint
from typing import List, Optional, TextIO

import numpy as np

from python_tsp.heuristics import solve_tsp_lin_kernighan
from python_tsp.utils import setup_initial_solution


def _print_message(
    msg: str, verbose: bool, log_file_handler: Optional[TextIO]
) -> None:
    if log_file_handler:
        print(msg, file=log_file_handler)

    if verbose:
        print(msg)


def solve_tsp_record_to_record(
    distance_matrix: np.ndarray,
    x0: Optional[List[int]] = None,
    max_iterations: Optional[int] = None,
    log_file: Optional[str] = None,
    verbose: bool = False,
):
    """
    Solve the traveling Salesperson Problem using a
    Record to Record iterative local search heuristic.

    Parameters
    ----------
    distance_matrix
        Distance matrix of shape (n x n) with the (i, j) entry indicating the
        distance from node i to j

    x0
        Initial permutation. If not provided, it starts with a random path.

    max_iterations
        The maximum number of iterations for the algorithm. If not specified,
        it defaults to the number of nodes in the distance matrix.

    log_file
        If not `None`, creates a log file with details about the whole
        execution.

    verbose
        If true, prints algorithm status every iteration.

    Returns
    -------
    Tuple
        A tuple containing the Hamiltonian cycle and its distance.

    References
    ----------
    Éric D. Taillard, "Design of Heuristic Algorithms for Hard Optimization,"
    Chapter 7, Problems of Chapter 7, 7.4 Record to Record, Springer, 2023.
    """
    n = distance_matrix.shape[0]
    max_iterations = max_iterations or n
    x, fx = setup_initial_solution(distance_matrix=distance_matrix, x0=x0)

    log_file_handler = (
        open(log_file, "w", encoding="utf-8") if log_file else None
    )

    for iteration in range(1, max_iterations + 1):
        xn = x[:]
        for _ in range(2):
            u = randint(1, n - 1)
            v = randint(1, n - 1)
            xn[u], xn[v] = xn[v], xn[u]

        xn, fn = solve_tsp_lin_kernighan(
            distance_matrix=distance_matrix, x0=xn
        )

        msg = f"Current value: {fx}; Iteration: {iteration}"
        _print_message(msg, verbose, log_file_handler)

        if fn < fx:
            x = xn[:]
            fx = fn

    if log_file_handler:
        log_file_handler.close()

    return x, fx
