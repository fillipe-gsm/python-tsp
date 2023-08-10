"""
``python_tsp.heuristics``
=========================
"""

from .local_search import solve_tsp_local_search  # noqa: F401
from .simulated_annealing import solve_tsp_simulated_annealing  # noqa: F401
from .variable_neighborhood_search import (  # noqa: F401
    solve_tsp_variable_neighborhood_search,
)
