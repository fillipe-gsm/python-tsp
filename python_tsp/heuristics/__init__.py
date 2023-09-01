"""
``python_tsp.heuristics``
=========================
"""

from .lin_kernighan import solve_tsp_lin_kernighan  # noqa: F401
from .local_search import solve_tsp_local_search  # noqa: F401
from .simulated_annealing import solve_tsp_simulated_annealing  # noqa: F401
