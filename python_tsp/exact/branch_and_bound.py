from typing import List, Tuple

import numpy as np

INF = np.iinfo(int).max


def compute_reduced_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Compute the reduced matrix and the total reduction cost.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix containing integer values.

    Returns
    -------
    Tuple[np.ndarray, int]
        A tuple containing the reduced matrix and the total reduction
        cost.

    Notes
    -----
    - The input matrix is expected to contain integer values, where INF
    (maximum integer value) represent an "infinite" value or an invalid
    entry in the matrix.
    - The reduction process involves subtracting the minimum value from
    each row and column of the matrix when the minimum value is not INF,
    effectively reducing the matrix size.
    - If a row or column contains only INF values, it is not considered
    in the reduction process.
    - The total reduction cost is the sum of the minimum values removed
    from the rows and columns during reduction.
    """
    mask = matrix != INF
    reduced_matrix = np.copy(matrix)

    min_rows = reduced_matrix.min(axis=1, keepdims=True)
    min_rows[min_rows == INF] = 0
    if np.any(min_rows != 0):
        reduced_matrix = np.where(
            mask, reduced_matrix - min_rows, reduced_matrix
        )

    min_cols = reduced_matrix.min(axis=0, keepdims=True)
    min_cols[min_cols == INF] = 0
    if np.any(min_cols != 0):
        reduced_matrix = np.where(
            mask, reduced_matrix - min_cols, reduced_matrix
        )

    return reduced_matrix, min_rows.sum() + min_cols.sum()


class Node:
    def __init__(self, parent_node, vertex) -> None:
        # stores the reduced matrix
        self.reduced_matrix: np.ndarray = None

        # stores node cost lower bound
        self.cost: int = INF

        # stores the current city number
        self.vertex: int = vertex

        # stores the tour path
        self.path: List[int] = []

        # stores the total number of cities visited so far
        self.level: int = 0

        # update path, level, reduced matrix and node cost
        # based on parent node.
        if parent_node:
            node_cost_matrix = np.copy(parent_node.reduced_matrix)
            node_cost_matrix[parent_node.vertex, :] = INF
            node_cost_matrix[:, self.vertex] = INF
            node_cost_matrix[self.vertex][0] = INF
            reduced_matrix, reduction_cost = compute_reduced_matrix(
                matrix=node_cost_matrix
            )
            self.reduced_matrix = reduced_matrix
            self.cost = (
                parent_node.cost
                + reduction_cost
                + parent_node.reduced_matrix[parent_node.vertex][self.vertex]
            )
            self.path = parent_node.path
            self.level = parent_node.level + 1

        self.path.append(self.vertex)

    def __lt__(self, other):
        return self.cost < other.cost


def solve_tsp_branch_and_bound(
    distance_matrix: np.ndarray,
) -> Tuple[List, float]:
    cost_matrix = np.copy(distance_matrix)
    cost_matrix[cost_matrix == 0] = INF
    return [], 0
