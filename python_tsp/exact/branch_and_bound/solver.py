from typing import List, Tuple

import numpy as np

from python_tsp.exact.branch_and_bound.node import Node
from python_tsp.exact.branch_and_bound.priority_queue import PriorityQueue


def solve_tsp_branch_and_bound(
    distance_matrix: np.ndarray,
) -> Tuple[List[int], float]:
    """
    Solve the Traveling Salesman Problem (TSP) using
    the Branch and Bound algorithm.

    Parameters
    ----------
    distance_matrix : numpy.ndarray
        The distance matrix representing the distances between cities.

    Returns
    -------
    Tuple[List[int], float]
        A tuple containing the optimal path (list of city indices) and its
        total cost. If the TSP cannot be solved, an empty path and cost of 0
        will be returned.
    """
    inf = np.iinfo(distance_matrix.dtype).max
    num_cities = len(distance_matrix)
    cost_matrix = np.copy(distance_matrix)
    np.fill_diagonal(cost_matrix, inf)

    root = Node.from_cost_matrix(cost_matrix=cost_matrix)
    pq = PriorityQueue([root])

    while not pq.empty:
        min_node = pq.pop()

        if min_node.level == num_cities - 1:
            return min_node.path, min_node.cost

        for index in range(num_cities):
            is_live_node = min_node.cost_matrix[min_node.index][index] != inf
            if is_live_node:
                live_node = Node.from_parent(parent=min_node, index=index)
                pq.push(live_node)

    return [], 0
