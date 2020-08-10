=================
Python TSP Solver
=================

``python-tsp`` is a library written in pure Python for solving typical Traveling
Salesperson Problems (TSP).

Examples
========
Suppose the following problem:

.. image:: figures/python_tsp_example.png

We can determine a Hamiltonian path with least cost simply using:

.. code:: python

   import numpy as np
   from python_tsp.exact import solve_tsp_dynamic_programming

   distance_matrix = np.array([
       [0,  5, 4, 10],
       [5,  0, 8,  5],
       [4,  8, 0,  3],
       [10, 5, 3,  0]
   ])
   permutation, distance = solve_tsp_dynamic_programming(distance_matrix)

The solution will be ``[0, 1, 3, 2]``, with total distance 17. Notice it is
always a closed path, so after node 2 we go back to 0.

If you opt for an open TSP version (it is not required to go back to the
origin), just make all elements of the first column of the distance matrix to
zero:

.. code:: python

   distance_matrix[:, 0] = 0
   permutation, distance = solve_tsp_dynamic_programming(distance_matrix)

and in this case we obtain ``[0, 2, 3, 1]``, with distance 12.

If you don't have a distance matrix, the ``distances`` module has functions to
compute an Euclidean distance matrix or the
`Great Circle Distance <https://en.wikipedia.org/wiki/Great-circle_distance>`_.

For example, if you have an array where each row has the latitude and longitude
of a point,

.. code:: python

   import numpy as np
   from python_tsp.distances import great_circle_distance_matrix

   sources = np.array([
       [ 40.73024833, -73.79440675],
       [ 41.47362495, -73.92783272],
       [ 41.26591   , -73.21026228],
       [ 41.3249908 , -73.507788  ]
   ])
   distance_matrix = great_circle_distance_matrix(sources)

Methods available
=================
There are two types of solvers available:

:Exact: Methods that always return the optimal solution of a problem.
        Use these solvers in relatively small instances (wherein "small" is
        relative to your requirements).

        - ``exact.solve_tsp_brute_force``: checks all permutations and returns
          the best one;

        - ``exact.solve_tsp_dynamic_programming``: uses a Dynamic Programming
          approach. It tends to be faster than the previous one, but it may
          demand more memory.

:Heuristics: These methods have no guarantees of finding the best solution,
             but usually return a good enough candidate in a more reasonable
             time for larger problems.

             - ``heuristics.solve_tsp_local_search``: local search heuristic.
               Fast, but it can get stuck in a local minimum;

             - ``heuristics.solve_tsp_simulated_annealing``: the Simulated
               Annealing metaheuristic. It may be slower, but it has better
               chances of avoiding getting trapped in local minima.


For developers
==============
The project uses `Python Poetry <https://python-poetry.org/>`_ to manage
dependencies. Check the website for installation instructions, or simply
install it with

.. code:: bash

   pip install poetry

After that, install all dependencies with ``poetry install``.

Here are the detailed steps that should be followed before making a pull
request:

.. code:: bash

  # Autopep8 and flake8 to be conformant with PEP8
  poetry run autopep8 --recursive --aggressive --in-place .
  poetry run flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
  poetry run flake8 . --count --exit-zero --max-complexity=10 \
  --max-line-length=79 --statistics
  # Mypy for proper type hints
  poetry run mypy --ignore-missing-imports .

You can also run all of these steps at once with the check-up bash script:

.. code:: bash

   bash ./.scripts/checkup_scripts.sh

Finally (and of course), make sure all tests pass:

.. code:: bash

   poetry run pytest tests/
