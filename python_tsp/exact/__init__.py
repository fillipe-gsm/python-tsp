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

    solve_tsp_brute_force
    solve_tsp_dynamic_programming


Tips on deciding the solver
---------------------------

In general, ``solve_tsp_dynamic_programming`` is faster, specially with an
appropriate `maxsize` input (the default is fine). However, because of its
recursion, it may take more memory, particularly if the number of nodes grows
large. If that becomes an issue and you still need a provably optimal solution,
use the ``solve_tsp_brute_force``.
"""

from .brute_force import solve_tsp_brute_force  # noqa
from .dynamic_programming import solve_tsp_dynamic_programming  # noqa
