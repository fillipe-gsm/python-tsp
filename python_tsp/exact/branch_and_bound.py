from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import List, Tuple

import numpy as np

INF = np.iinfo(int).max


@dataclass
class Node:
    """
    Node class representing a node in a graph traversal algorithm.

    Attributes
    ----------
    level : int
        The level or depth of the node in the graph traversal.

    index : int
        The index or vertex associated with the node.

    path : List[int]
        The path to the node, represented as a list of vertex indices.

    cost : int
        The cost or evaluation value of the node, used for
        priority in the traversal.

    reduced_matrix : np.ndarray
        The reduced matrix representing the state of the graph after reduction
        operations during traversal.

    Methods
    -------
    compute_reduced_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, int]:
        Compute the reduced matrix and the total reduction cost.

    root_from_cost_matrix(cost_matrix: np.ndarray) -> Node:
        Create the root node from the cost matrix using reduction.

    from_parent_node(parent_node: Node, node_index: int) -> Node:
        Create a new node based on a parent node and a specified index.

    __lt__(self: Node, other: Node) -> bool:
        Compare nodes based on their cost values.
    """

    level: int
    index: int
    path: List[int]
    cost: int
    reduced_matrix: np.ndarray

    @staticmethod
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

        min_rows = np.min(reduced_matrix, axis=1, keepdims=True)
        min_rows[min_rows == INF] = 0
        if np.any(min_rows != 0):
            reduced_matrix = np.where(
                mask, reduced_matrix - min_rows, reduced_matrix
            )

        min_cols = np.min(reduced_matrix, axis=0, keepdims=True)
        min_cols[min_cols == INF] = 0
        if np.any(min_cols != 0):
            reduced_matrix = np.where(
                mask, reduced_matrix - min_cols, reduced_matrix
            )

        return reduced_matrix, min_rows.sum() + min_cols.sum()

    @classmethod
    def root_from_cost_matrix(
        cls,
        cost_matrix: np.ndarray,
    ) -> Node:
        """
        Create the root node from the cost matrix using reduction.

        Parameters
        ----------
        cost_matrix : np.ndarray
            The input cost matrix containing integer values.

        Returns
        -------
        Node
            The root node with the computed reduction and cost.

        Notes
        -----
        This method creates the root node for the graph traversal by performing
        the following steps:
        1. Compute the reduced matrix and the total reduction cost using the
           `compute_reduced_matrix` static method.
        2. Initialize the root node with level 0, index 0, and the computed
           reduction cost.
        3. Set the reduced matrix and path attributes of the root node based
           on the computed values.
        4. The path of the root node is initialized with only the starting
           vertex, which is 0.

        The root node represents the starting point of the graph traversal and
        serves as the initial state for the search algorithm.
        """
        _reduced_matrix, _reduction_cost = cls.compute_reduced_matrix(
            matrix=cost_matrix
        )
        return cls(
            level=0,
            index=0,
            path=[0],
            cost=_reduction_cost,
            reduced_matrix=_reduced_matrix,
        )

    @classmethod
    def from_parent_node(
        cls,
        parent_node: Node,
        node_index: int,
    ) -> Node:
        """
        Create a new node based on a parent node and a specified index.

        Parameters
        ----------
        parent_node : Node
            The parent node from which to create the new node.
        node_index : int
            The index or vertex associated with the new node.

        Returns
        -------
        Node
            A new node based on the parent node and the specified index.

        Notes
        -----
        This method creates a new node by performing the following steps:
        1. Create a copy of the parent node's reduced matrix.
        2. Set all entries in the row and column corresponding to the parent
           node's index to "infinite" (INF) to prevent revisiting the same
           vertex.
        3. Set the cost of reaching the new node as the sum of the parent
           node's cost, the reduction cost of the modified matrix, and the cost
           of moving from the parent node's index to the new node's index.
        4. Create a new path for the new node by copying the parent node's path
           and appending the new node's index.

        This method is used to explore the graph and generate new nodes
        during a graph traversal algorithm.
        """
        _cost_matrix = np.copy(parent_node.reduced_matrix)
        _cost_matrix[parent_node.index, :] = INF
        _cost_matrix[:, node_index] = INF
        _cost_matrix[node_index][0] = INF
        _reduced_matrix, _reduction_cost = cls.compute_reduced_matrix(
            matrix=_cost_matrix
        )
        _cost = (
            parent_node.cost
            + _reduction_cost
            + parent_node.reduced_matrix[parent_node.index][node_index]
        )
        return cls(
            level=parent_node.level + 1,
            index=node_index,
            path=parent_node.path[:] + [node_index],
            cost=_cost,
            reduced_matrix=_reduced_matrix,
        )

    def __lt__(self: Node, other: Node):
        """
        Compare nodes based on their cost values.

        Parameters
        ----------
        other : Node
            Another node to compare with.

        Returns
        -------
        bool
            True if the current node's cost is less than the other node's cost,
            False otherwise.
        """
        return self.cost < other.cost

    def __repr__(self: Node):
        return f"Node(index={self.index}, cost={self.cost})"


@dataclass
class NodePriorityQueue:
    """
    Priority Queue implementation for managing Node objects.

    Attributes
    ----------
    _container : List[Node]
        The internal container to store the Node objects in a
        heap-based priority queue. Default is an empty list.

    Methods
    -------
    empty() -> bool:
        Check if the priority queue is empty.

    push(item: Node) -> None:
        Push a Node object into the priority queue with
        respect to its priority.

    pop() -> Node:
        Pop the Node object with the highest priority from the priority queue.
    """

    _container: List[Node] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self._container

    def push(self, item: Node) -> None:
        heappush(self._container, item)

    def pop(self) -> Node:
        return heappop(self._container)


def solve_tsp_branch_and_bound(
    distance_matrix: np.ndarray,
) -> Tuple[List[int], float]:
    """
    Solve the Traveling Salesman Problem (TSP) using
    the Branch and Bound algorithm.

    Parameters
    ----------
    distance_matrix : np.ndarray
        The distance matrix containing the distances between cities.

    Returns
    -------
    Tuple[List[int], float]
        A tuple containing the optimal path as a list of city
        indices and the total cost of the path.

    Notes
    -----
    The Branch and Bound algorithm is used to find the optimal path for
    the Traveling Salesman Problem, which aims to find the shortest
    possible route that visits each city exactly once and returns to the
    starting city.

    The distance matrix should contain the distances between cities. If
    there is no direct connection between two cities, the distance value
    should be set to INF.

    The algorithm works as follows:
    - Convert the distance matrix into a cost matrix by replacing zeros with
      INF.
    - Initialize a priority queue to store nodes (partial
      paths) for exploration, starting with the root node.
    - Continuously expand nodes with the lowest cost from the priority
      queue.
    - If a complete path (tour) is found (i.e., all cities are
      visited), return the optimal path and its cost.
    - Generate child nodes for the current node by considering valid
      connections between cities.
    - Add child nodes to the priority queue for further exploration.
    - Continue until all nodes are explored or the optimal path is found.

    If no feasible path is found, an empty list and cost 0 are returned.
    """
    cost_matrix = np.where(distance_matrix == 0, INF, distance_matrix)
    num_cities = len(cost_matrix)
    node_indices = range(num_cities)
    pq = NodePriorityQueue()
    root = Node.root_from_cost_matrix(cost_matrix=cost_matrix)
    pq.push(root)

    while not pq.empty:
        min_cost_node = pq.pop()

        if min_cost_node.level == num_cities - 1:
            return min_cost_node.path, min_cost_node.cost

        for node_index in node_indices:
            is_live_node = (
                min_cost_node.reduced_matrix[min_cost_node.index][node_index]
                != INF
            )
            if is_live_node:
                live_node = Node.from_parent_node(
                    parent_node=min_cost_node, node_index=node_index
                )
                pq.push(live_node)

    return [], 0
