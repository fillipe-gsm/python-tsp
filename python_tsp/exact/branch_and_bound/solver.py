from math import inf
from typing import List, Tuple

import numpy as np

from python_tsp.exact.branch_and_bound import Node, PriorityQueue


def solve_tsp_branch_and_bound(
    distance_matrix: np.ndarray,
) -> Tuple[List[int], float]:
    """
    Solve the Traveling Salesperson Problem (TSP) using the
    Branch and Bound algorithm.

    Parameters
    ----------
    distance_matrix
        The distance matrix representing the distances between cities.

    Returns
    -------
    Tuple
        A tuple containing the optimal path (list of city indices) and its
        total cost. If the TSP cannot be solved, an empty path and a cost
        of positive infinity will be returned.

    Notes
    -----
    The `distance_matrix` should be a square matrix with non-negative
    values. The element `distance_matrix[i][j]` represents the distance from
    city `i` to city `j`. If two cities are not directly connected, the
    distance should be set to a float value of positive infinity
    (float('inf')).

    The path is represented as a list of city indices, and the total cost is a
    float value indicating the sum of distances in the optimal path.

    If the TSP cannot be solved (e.g., due to disconnected cities), the
    function will return an empty path ([]) and a cost of positive infinity
    (float('inf')).

    References
    ----------
    .. [1] Horowitz, E., Sahni, S., & Rajasekaran, S. (1997).
           Computer Algorithms. Chapter 8 - Branch and Bound. Section 8.3.
           W. H. Freeman and Company.
    """
    num_cities = len(distance_matrix)
    cost_matrix = np.copy(distance_matrix).astype(float)
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

    return [], inf
