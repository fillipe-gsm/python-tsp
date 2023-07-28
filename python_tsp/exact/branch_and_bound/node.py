from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Node:
    """
    Represents a node in the search tree for the Traveling Salesman Problem.

    Attributes
    ----------
    level : int
        The level of the node in the search tree.
    index : int
        The index of the current city in the path.
    path : List[int]
        The list of city indices visited so far.
    cost : int
        The total cost of the path up to this node.
    cost_matrix : numpy.ndarray
        The cost matrix representing the distances between cities.

    Methods
    -------
    compute_reduced_matrix(matrix: numpy.ndarray) -> Tuple[numpy.ndarray, int]:
        Compute the reduced matrix and the cost of reducing it.
    from_cost_matrix(cost_matrix: numpy.ndarray) -> Node:
        Create a Node object from a given cost matrix.
    from_parent(parent: Node, index: int) -> Node:
        Create a new Node object based on a parent node and a city index.
    """

    level: int
    index: int
    path: List[int]
    cost: int
    cost_matrix: np.ndarray

    @staticmethod
    def compute_reduced_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Compute the reduced matrix and the cost of reducing it.

        Parameters
        ----------
        matrix : numpy.ndarray
            The cost matrix to compute the reductions.

        Returns
        -------
        Tuple[numpy.ndarray, int]
            A tuple containing the reduced matrix and the total
            cost of reductions.
        """
        inf = np.iinfo(matrix.dtype).max
        mask = matrix != inf
        reduced_matrix = np.copy(matrix)

        min_rows = np.min(reduced_matrix, axis=1, keepdims=True)
        min_rows[min_rows == inf] = 0
        if np.any(min_rows != 0):
            reduced_matrix = np.where(
                mask, reduced_matrix - min_rows, reduced_matrix
            )

        min_cols = np.min(reduced_matrix, axis=0, keepdims=True)
        min_cols[min_cols == inf] = 0
        if np.any(min_cols != 0):
            reduced_matrix = np.where(
                mask, reduced_matrix - min_cols, reduced_matrix
            )

        return reduced_matrix, min_rows.sum() + min_cols.sum()

    @classmethod
    def from_cost_matrix(cls, cost_matrix: np.ndarray) -> Node:
        """
        Create a Node object from a given cost matrix.

        Parameters
        ----------
        cost_matrix : numpy.ndarray
            The cost matrix representing the distances between cities.

        Returns
        -------
        Node
            A new Node object initialized with the reduced cost matrix.
        """
        _cost_matrix, _cost = cls.compute_reduced_matrix(matrix=cost_matrix)
        return cls(
            level=0,
            index=0,
            path=[0],
            cost=_cost,
            cost_matrix=_cost_matrix,
        )

    @classmethod
    def from_parent(cls, parent: Node, index: int) -> Node:
        """
        Create a new Node object based on a parent node and a city index.

        Parameters
        ----------
        parent : Node
            The parent node.
        index : int
            The index of the new city to be added to the path.

        Returns
        -------
        Node
            A new Node object with the updated path and cost.
        """
        matrix = np.copy(parent.cost_matrix)
        inf = np.iinfo(matrix.dtype).max
        matrix[parent.index, :] = inf
        matrix[:, index] = inf
        matrix[index][0] = inf
        _cost_matrix, _cost = cls.compute_reduced_matrix(matrix=matrix)
        return cls(
            level=parent.level + 1,
            index=index,
            path=parent.path[:] + [index],
            cost=(
                parent.cost + _cost + parent.cost_matrix[parent.index][index]
            ),
            cost_matrix=_cost_matrix,
        )

    def __lt__(self: Node, other: Node):
        """
        Compare two Node objects based on their costs.

        Parameters
        ----------
        other : Node
            The other Node object to compare with.

        Returns
        -------
        bool
            True if this Node's cost is less than the other Node's
            cost, False otherwise.
        """
        return self.cost < other.cost
