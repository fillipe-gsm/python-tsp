=========
Distances
=========

In order to solve a TSP, one of the most important input parameters is a distance matrix. However, it may be the case that the user has only the node coordinates, or maybe the target is a TSPLIB file. To help handling with that, this library has some distance functions implemented.

Euclidean Distance
==================

Computes the regular Euclidean distance between points.

.. code:: python

    import numpy as np

    from python_tsp.distances import euclidean_distance_matrix


    sources = np.array([[0, 0], [1, 1]])
    destinations = np.array([[2, 2], [3, 3], [4, 4]])

    distance_matrix = euclidean_distance_matrix(sources, destinations)
    # outputs a 2 x 3 numpy array


The returned distance matrix has in the ``i``-th row the distance from the ``i``-th source to each destination. The API is similar to all distance functions here.

Notice that, in general, the distance matrix is non-square. While this is by design to allow its use in more generic situations, for the TSP we need the distance between each point, so it must be square. Thus, either set ``destinations`` equal to ``sources`` or leave the second unset:

.. code:: python

    distance_matrix = euclidean_distance_matrix(sources)  # a 2 x 2 array
    # same as
    # distance_matrix = euclidean_distance_matrix(sources, sources)
