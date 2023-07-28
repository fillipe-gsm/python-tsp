from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import List

from python_tsp.exact.branch_and_bound.node import Node


@dataclass
class PriorityQueue:
    """
    A priority queue implementation using a binary heap
    for efficient element retrieval.

    Attributes
    ----------
    _container : List[Node]
        The list that holds the elements in the priority queue.

    Methods
    -------
    empty() -> bool:
        Check if the priority queue is empty.
    push(item: Node) -> None:
        Push an item into the priority queue.
    pop() -> Node:
        Pop the item with the highest priority from the priority queue.
    """

    _container: List[Node] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        """
        Check if the priority queue is empty.

        Returns
        -------
        bool
            True if the priority queue is empty, False otherwise.
        """
        return not self._container

    def push(self, item: Node) -> None:
        """
        Push an item into the priority queue.

        Parameters
        ----------
        item : Node
            The item to be pushed into the priority queue.

        Returns
        -------
        None
        """
        heappush(self._container, item)

    def pop(self) -> Node:
        """
        Pop the item with the highest priority from the priority queue.

        Returns
        -------
        Node
            The node with the highest priority.
        """
        return heappop(self._container)
