"""
``python_tsp.exact``
====================

Module with exact solvers, which have a guarantee of returning the optimal
solution of a TSP problem. They may be good options for small scale problems
that need to be solved many times in a row, even more if when calling other
solvers can add too much of an overhead in implementing a TSP model, calling a
wrapper from another language etc.

It is up to the user to decide how small enough is a "small scale problem". We
suggest to test the solvers with toy problems with typical number of nodes
required and check if the processing time is acceptable.

Functions present in this module are listed below.

    solve_tsp_branch_and_bound
    solve_tsp_brute_force
    solve_tsp_dynamic_programming


Tips on deciding the solver
---------------------------

The choice among the three exact methods depends on the specific
characteristics of the Traveling Salesperson Problem (TSP) you are
dealing with:

If the TSP has only a few cities and the goal is a quick solution without
worrying about scalability, ``solve_tsp_brute_force`` may be a simple
and viable choice, but only for educational purposes or small cases.
If the TSP is relatively small (with a few cities) and precision is
essential, ``solve_tsp_dynamic_programming`` may be preferable, as
long as the required memory and execution time are not prohibitive.
If the TSP has many cities and an exact solution is required,
``solve_tsp_branch_and_bound`` is more scalable and, therefore, more
suitable for such scenarios.

In general, ``solve_tsp_brute_force`` is not recommended for TSPs of
significant size due to its exponential complexity.
``solve_tsp_dynamic_programming`` and ``solve_tsp_branch_and_bound``
are more efficient approaches to finding the optimal solution, but the
choice between them will depend on the problem size and available
computational resources.
"""

from .branch_and_bound import solve_tsp_branch_and_bound  # noqa: F401
from .brute_force import solve_tsp_brute_force  # noqa: F401
from .dynamic_programming import solve_tsp_dynamic_programming  # noqa: F401
