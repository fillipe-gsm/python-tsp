from typing import List, Optional, TextIO, Tuple

import numpy as np

from python_tsp.exact import solve_tsp_brute_force
from python_tsp.utils import setup_initial_solution


def _cycle_to_successors(cycle: List[int]) -> List[int]:
    """
    Convert a cycle representation to successors representation.

    Parameters
    ----------
    cycle
        A list representing a cycle.

    Returns
    -------
    List
        A list representing successors.
    """
    successors = cycle[:]
    n = len(cycle)
    for i, _ in enumerate(cycle):
        successors[cycle[i]] = cycle[(i + 1) % n]
    return successors


def _successors_to_cycle(successors: List[int]) -> List[int]:
    """
    Convert a successors representation to a cycle representation.

    Parameters
    ----------
    successors
        A list representing successors.

    Returns
    -------
    List
        A list representing a cycle.
    """
    cycle = successors[:]
    j = 0
    for i, _ in enumerate(successors):
        cycle[i] = j
        j = successors[j]
    return cycle


def _minimizes_hamiltonian_path_distance(
    tabu: np.ndarray,
    iteration: int,
    successors: List[int],
    ejected_edge: Tuple[int, int],
    distance_matrix: np.ndarray,
    hamiltonian_path_distance: float,
    hamiltonian_cycle_distance: float,
) -> Tuple[int, int, float]:
    """
    Minimize the Hamiltonian path distance after ejecting an edge.

    Parameters
    ----------
    tabu
        A NumPy array for tabu management.

    iteration
        The current iteration.

    successors
        A list representing successors.

    ejected_edge
        The edge that was ejected.

    distance_matrix
        A NumPy array representing the distance matrix.

    hamiltonian_path_distance
        The Hamiltonian path distance.

    hamiltonian_cycle_distance
        The Hamiltonian cycle distance.

    Returns
    -------
    Tuple
        The best c, d, and the new Hamiltonian path distance found.
    """
    a, b = ejected_edge
    best_c = c = last_c = successors[b]
    path_cb_distance = distance_matrix[c, b]
    path_bc_distance = distance_matrix[b, c]
    hamiltonian_path_distance_found = hamiltonian_cycle_distance

    while successors[c] != a:
        d = successors[c]
        path_cb_distance += distance_matrix[c, last_c]
        path_bc_distance += distance_matrix[last_c, c]
        new_hamiltonian_path_distance_found = (
            hamiltonian_path_distance
            + distance_matrix[b, d]
            - distance_matrix[c, d]
            + path_cb_distance
            - path_bc_distance
        )

        if (
            new_hamiltonian_path_distance_found + distance_matrix[a, c]
            < hamiltonian_cycle_distance
        ):
            return c, d, new_hamiltonian_path_distance_found

        if (
            tabu[c, d] != iteration
            and new_hamiltonian_path_distance_found
            < hamiltonian_path_distance_found
        ):
            hamiltonian_path_distance_found = (
                new_hamiltonian_path_distance_found
            )
            best_c = c

        last_c = c
        c = d

    return best_c, successors[best_c], hamiltonian_path_distance_found


def _print_message(
    msg: str, verbose: bool, log_file_handler: Optional[TextIO]
) -> None:
    if log_file_handler:
        print(msg, file=log_file_handler)

    if verbose:
        print(msg)


def _solve_tsp_brute_force(
    distance_matrix: np.ndarray,
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[List[int], float]:
    x, fx = solve_tsp_brute_force(distance_matrix)
    x = x or []

    log_file_handler = (
        open(log_file, "w", encoding="utf-8") if log_file else None
    )
    msg = (
        "Few nodes to use Lin-Kernighan heuristics, "
        "using Brute Force instead. "
    )
    if not x:
        msg += "No solution found."
    else:
        msg += f"Found value: {fx}"
    _print_message(msg, verbose, log_file_handler)

    if log_file_handler:
        log_file_handler.close()

    return x, fx


def solve_tsp_lin_kernighan(
    distance_matrix: np.ndarray,
    x0: Optional[List[int]] = None,
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[List[int], float]:
    """
    Solve the Traveling Salesperson Problem using the Lin-Kernighan algorithm.

    Parameters
    ----------
    distance_matrix
        Distance matrix of shape (n x n) with the (i, j) entry indicating the
        distance from node i to j

    x0
        Initial permutation. If not provided, it starts with a random path.

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
    Chapter 5, Section 5.3.2.1: Lin-Kernighan Neighborhood, Springer, 2023.
    """
    num_vertices = distance_matrix.shape[0]
    if num_vertices < 4:
        return _solve_tsp_brute_force(distance_matrix, log_file, verbose)

    hamiltonian_cycle, hamiltonian_cycle_distance = setup_initial_solution(
        distance_matrix=distance_matrix, x0=x0
    )
    vertices = list(range(num_vertices))
    iteration = 0
    improvement = True
    tabu = np.zeros(shape=(num_vertices, num_vertices), dtype=int)

    log_file_handler = (
        open(log_file, "w", encoding="utf-8") if log_file else None
    )

    while improvement:
        iteration += 1
        improvement = False
        successors = _cycle_to_successors(hamiltonian_cycle)

        # Eject edge [a, b] to start the chain and compute the Hamiltonian
        # path distance obtained by ejecting edge [a, b] from the cycle
        # as reference.
        a = int(distance_matrix[vertices, successors].argmax())
        b = successors[a]
        hamiltonian_path_distance = (
            hamiltonian_cycle_distance - distance_matrix[a, b]
        )

        while True:
            ejected_edge = a, b

            # Find the edge [c, d] that minimizes the Hamiltonian path obtained
            # by removing edge [c, d] and adding edge [b, d], with [c, d] not
            # removed in the current ejection chain.
            (
                c,
                d,
                hamiltonian_path_distance_found,
            ) = _minimizes_hamiltonian_path_distance(
                tabu,
                iteration,
                successors,
                ejected_edge,
                distance_matrix,
                hamiltonian_path_distance,
                hamiltonian_cycle_distance,
            )

            # If the Hamiltonian cycle cannot be improved, return
            # to the solution and try another ejection.
            if hamiltonian_path_distance_found >= hamiltonian_cycle_distance:
                break

            # Update Hamiltonian path distance reference
            hamiltonian_path_distance = hamiltonian_path_distance_found

            # Reverse the direction of the path from b to c
            i, si, successors[b] = b, successors[b], d
            while i != c:
                successors[si], i, si = i, si, successors[si]

            # Don't remove again the minimal edge found
            tabu[c, d] = tabu[d, c] = iteration

            # c plays the role of b in the next iteration
            b = c

            msg = (
                f"Current value: {hamiltonian_cycle_distance}; "
                f"Ejection chain: {iteration}"
            )
            _print_message(msg, verbose, log_file_handler)

            # If the Hamiltonian cycle improves, update the solution
            if (
                hamiltonian_path_distance + distance_matrix[a, b]
                < hamiltonian_cycle_distance
            ):
                improvement = True
                successors[a] = b
                hamiltonian_cycle = _successors_to_cycle(successors)
                hamiltonian_cycle_distance = (
                    hamiltonian_path_distance + distance_matrix[a, b]
                )

    if log_file_handler:
        log_file_handler.close()

    return hamiltonian_cycle, hamiltonian_cycle_distance
