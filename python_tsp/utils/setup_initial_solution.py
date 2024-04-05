from random import sample, choice
from typing import List, Optional, Tuple

import numpy as np

from .permutation_distance import compute_permutation_distance


STARTING_NODE_OUTSIDE_BOUNDARIES_MSG = (
    "Starting node outside limits [0, num_nodes]"
)
ENDING_NODE_OUTSIDE_BOUNDARIES_MSG = (
    "Ending node outside limits [0, num_nodes]"
)
STARTING_ENDING_NODE_ARE_EQUAL_MSG = (
    "Starting and ending nodes cannot be equal"
)
INVALID_SIZE_INITIAL_SOLUTION_MSG = (
    "`x0` size does not match the number of nodes from distance matrix"
)
INVALID_INITIAL_SOLUTION_MSG = (
    "`x0` does not contain a permutation of all nodes"
)
OVERLAPPING_INPUT_ARGUMENTS_MSG = (
    "Cannot set `starting_node` or `ending_node` if `x0` is provided"
)


def setup_initial_solution(
    distance_matrix: np.ndarray,
    x0: Optional[List[int]] = None,
    starting_node: Optional[int] = None,
    ending_node: Optional[int] = None,
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

    starting_node
        First node to appear in the permutation.

    ending_node
        Last node to appear in the permutation. Note, despite the name, the
        TSP is still by definition a cycle, so we always go back from
        `ending_node` to `starting_node`.

    Returns
    -------
    x0
        Permutation with initial solution. If ``x0`` was provided, it is the
        same list

    fx0
        Objective value of x0
    """
    _validate_input_arguments(distance_matrix, x0, starting_node, ending_node)

    if not x0:
        num_nodes = distance_matrix.shape[0]
        x0 = _build_initial_permutation(
            num_nodes, starting_node=starting_node, ending_node=ending_node
        )

    fx0 = compute_permutation_distance(distance_matrix, x0)
    return x0, fx0


def _validate_input_arguments(
    distance_matrix: np.ndarray,
    x0: Optional[List[int]],
    starting_node: Optional[int],
    ending_node: Optional[int],
) -> None:
    """Validate combination of input parameters"""
    num_nodes = distance_matrix.shape[0]

    if x0:
        all_nodes = set(range(num_nodes))
        if len(x0) != num_nodes:
            raise ValueError(INVALID_SIZE_INITIAL_SOLUTION_MSG)
        if set(x0) != all_nodes:
            raise ValueError(INVALID_INITIAL_SOLUTION_MSG)
        if starting_node is not None or ending_node is not None:
            raise ValueError(OVERLAPPING_INPUT_ARGUMENTS_MSG)

    if starting_node is not None:
        if starting_node < 0 or starting_node >= num_nodes:
            raise ValueError(STARTING_NODE_OUTSIDE_BOUNDARIES_MSG)

    if ending_node is not None:
        if ending_node < 0 or ending_node >= num_nodes:
            raise ValueError(ENDING_NODE_OUTSIDE_BOUNDARIES_MSG)

    if starting_node is not None and ending_node is not None:
        if starting_node == ending_node:
            raise ValueError(STARTING_ENDING_NODE_ARE_EQUAL_MSG)


def _build_initial_permutation(
    num_nodes: int,
    starting_node: Optional[int] = None,
    ending_node: Optional[int] = None,
) -> List[int]:
    """
    Build a random list of integers from 0 to `num_nodes` - 1 guaranteeing the
    initial node is `starting_node` and the last one is `ending_node`.

    If not provided, `starting_node` is assumed as 0 and `ending_node` is taken
    randomly.
    """

    starting_node = starting_node or 0

    all_nodes_except_starting_node = [
        node for node in range(num_nodes) if node != starting_node
    ]
    ending_node = (
        ending_node
        if ending_node is not None
        else choice(all_nodes_except_starting_node)
    )

    all_nodes_without_extremes = [
        node
        for node in range(num_nodes)
        if node != starting_node and node != ending_node
    ]
    x0 = (
        [starting_node]
        + sample(all_nodes_without_extremes, len(all_nodes_without_extremes))
        + [ending_node]
    )

    return x0
