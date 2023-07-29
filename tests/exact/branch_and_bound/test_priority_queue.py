import numpy as np
from math import inf
from python_tsp.exact.branch_and_bound import Node, PriorityQueue


def test_priority_queue():
    """
    Test the `PriorityQueue` class.

    Verifies the functionality of the priority queue implementation.

    The priority queue is initialized with a root node created from a cost
    matrix. Then, live nodes are created from the root node by adding
    neighbors one by one. The test checks if the priority queue correctly
    handles pushing and popping nodes.

    Verifications:
        - The priority queue should not be empty after pushing nodes.
        - The cost of the node popped from the priority queue should be 25.
    """
    cost_matrix = np.array(
        [
            [inf, 20, 30, 10, 11],
            [15, inf, 16, 4, 2],
            [3, 5, inf, 2, 4],
            [19, 6, 18, inf, 3],
            [16, 4, 7, 16, inf],
        ]
    )

    root = Node.from_cost_matrix(cost_matrix=cost_matrix)
    pq = PriorityQueue([root])

    for index in range(len(cost_matrix)):
        if root.cost_matrix[root.index][index] != inf:
            live_node = Node.from_parent(parent=root, index=index)
            pq.push(live_node)

    assert not pq.empty
    assert pq.pop().cost == 25
