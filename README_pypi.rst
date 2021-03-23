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

Given a distance matrix as a numpy array, it is easy to compute a Hamiltonian
path with least cost. For instance, to use a Dynamic Programming method:

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

To solve the same problem with a metaheuristic method:

.. code:: python

   from python_tsp.heuristics import solve_tsp_simulated_annealing

   permutation, distance = solve_tsp_simulated_annealing(distance_matrix) 

Keep in mind that, being a metaheuristic, the solution may vary from execution
to execution, and there is no guarantee of optimality. However, it may be a
way faster alternative in larger instances.

If you with for an open TSP version (it is not required to go back to the
origin), just set all elements of the first column of the distance matrix to
zero:

.. code:: python

   distance_matrix[:, 0] = 0
   permutation, distance = solve_tsp_dynamic_programming(distance_matrix)

and in this case we obtain ``[0, 2, 3, 1]``, with distance 12. Notice that in
this case the distance matrix is actually asymmetric, and the methods here are
applicable as well.

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

See the `project's repository <https://github.com/fillipe-gsm/python-tsp>`_ 
for more examples and a list of available methods.
