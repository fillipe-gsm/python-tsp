=================
Python TSP Solver
=================

``python-tsp`` is a library written in pure Python for solving typical Traveling
Salesperson Problems (TSP). It can work with symmetric and asymmetric versions.


Installation
============
.. code:: bash

  pip install python-tsp


Examples
========

Regular TSP problem
-------------------

Suppose we wish to find a Hamiltonian path with least cost for the following 
problem:

.. image:: figures/python_tsp_example.png

We can find an optimal path using a Dynamic Programming method with:

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

There are also heuristic-based approaches to solve the same problem. For
instance, to use a local search method:

.. code:: python

   from python_tsp.heuristics import solve_tsp_local_search

   permutation, distance = solve_tsp_local_search(distance_matrix)

In this case there is generally no guarantee of optimality, but in this small
instance the answer is normally a permutation with total distance 17 as well
(notice in this case there are many permutations with the same optimal
distance).

Open TSP problem
----------------

If you opt for an open TSP version (it is not required to go back to the
origin), just set all elements of the first column of the distance matrix to
zero:

.. code:: python

   distance_matrix[:, 0] = 0
   permutation, distance = solve_tsp_dynamic_programming(distance_matrix)

and in this case we obtain ``[0, 2, 3, 1]``, with distance 12. Notice that in
this case the distance matrix is actually asymmetric, and the methods here are
applicable as well.


Computing a distance matrix
---------------------------

The previous examples assumed you already had a distance matrix. If that is not
the case, the ``distances`` module has prepared some functions to compute an 
Euclidean distance matrix or a
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

After that, clone the repo and install dependencies with ``poetry install``.

Here are the detailed steps that should be followed before making a pull
request:

.. code:: bash

  # Autopep8 and flake8 to be conformant with PEP8
  poetry run autopep8 --recursive --aggressive --in-place .
  poetry run flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
  poetry run flake8 . --count --exit-zero --max-complexity=10 --max-line-length=79 --statistics

  # Mypy for proper type hints
  poetry run mypy --ignore-missing-imports .

You can also run all of these steps at once with the check-up bash script:

.. code:: bash

   ./.scripts/checkup_scripts.sh
   bash ./.scripts/checkup_scripts.sh  # if the previous one fails

Finally (and of course), make sure all tests pass and you get at least 95% of
coverage:

.. code:: bash

  poetry run pytest --cov=. --cov-report=term-missing --cov-fail-under=95 tests/


Release Notes and Changelog
===========================

=======
Release 0.1.2
-------------
- Local search and Simulated Annealing random solution now begins at root node
  0 just like the exact methods.

Python support:

* Python >= 3.6

Release 0.1.1
-------------

Improved Python versions support.

Python support:

* Python >= 3.6


Release 0.1.0
-------------

Initial version. Support for the following solvers:

* Exact (Brute force and Dynamic Programming);
* Heuristics (Local Search and Simulated Annealing).

The local search-based algorithms can be run with neighborhoods PS1, PS2 and
PS3.

Python support:

* Python >= 3.8
