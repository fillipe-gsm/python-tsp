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


The returned distance matrix has in the ``i``-th row the distance from the ``i``-th source to each destination. This API is similar to all distance functions here.

Notice that, in general, the distance matrix is non-square. While this is by design to allow its use in more generic situations, for the TSP we need the distance between each point, so it must be square. Thus, either set ``destinations`` equal to ``sources`` or leave the second unset:

.. code:: python

    distance_matrix = euclidean_distance_matrix(sources)  # a 2 x 2 array
    # same as
    # distance_matrix = euclidean_distance_matrix(sources, sources)

If your problem requires a matrix of integers, just manipulate it like any Numpy array; e.g., ``distance_matrix.astype(int)``.


Great Circle Distance
=====================

In case the nodes represent coordinates in a sphere (such as the planet Earth), a more appropriate distance can be the `Great Circle Distance <https://en.wikipedia.org/wiki/Great-circle_distance>`_.

For example, if you have an array where each row has the latitude and longitude of a point,

.. code:: python

   import numpy as np

   from python_tsp.distances import great_circle_distance_matrix


   sources = np.array([
       [ 40.73024833, -73.79440675],  # latitude, longitude
       [ 41.47362495, -73.92783272],
       [ 41.26591   , -73.21026228],
       [ 41.3249908 , -73.507788  ]
   ])
   distance_matrix = great_circle_distance_matrix(sources)


Notice that the order of the coordinates must be "latitude, longitude". Also, while the Euclidean distance has no real dimension, the distance matrix here is returned in *meters*.


Street Distance (OpenStreetMaps)
================================

Again in case you have coordinates but would like to take a city's geography into consideration, you can compute street distances via an Open Source Routing Machine (`OSRM <http://project-osrm.org/>`_) server.


.. code:: python

   import numpy as np

   from python_tsp.distances import osrm_distance_matrix


   sources = np.array([
       [ 40.73024833, -73.79440675],  # latitude, longitude
       [ 41.47362495, -73.92783272],
       [ 41.26591   , -73.21026228],
       [ 41.3249908 , -73.507788  ]
   ])
   distance_matrix = osrm_distance_matrix(
       sources, osrm_server_address="http://localhost:5000"
   )

The function will format the input and perform a request to the server, outputting the street distance matrix portion.

Notice this requires an OSRM server running that can be accessed. This is typically done locally with docker containers (as shown in the `documentation <https://github.com/Project-OSRM/osrm-backend#using-docker>`_). In the previous example, the server would be running locally at port 5000, which is as shown in the docs.

If your input is small and you don't feel like going through all steps just for that, you can use the public server like:


.. code:: python

   distance_matrix = osrm_distance_matrix(
       sources, osrm_server_address="http://router.project-osrm.org"
   )

It is also possible to send request batches to prevent errors with max table constraints:

.. code:: python

   distance_matrix = osrm_distance_matrix(
       sources,
       osrm_server_address="http://router.project-osrm.org",
       osrm_batch_size=50,
   )


This sends multiple requests with at most 50 nodes at a time.

Also, despite "distances" in the name, it is also possible to output the *duration* matrix between the nodes:


.. code:: python

   distance_matrix = osrm_distance_matrix(
       sources,
       osrm_server_address="http://router.project-osrm.org",
       osrm_batch_size=50,
       cost_type="durations",  # "distances" is the default
   )

Finally, remember you can also compute the distance between different sources and destinations:

.. code:: python

    sources = np.array([
       [ 40.73024833, -73.79440675],  # latitude, longitude
       [ 41.47362495, -73.92783272],
       [ 41.26591   , -73.21026228],
    ])
    destinations = np.array([
       [ 41.3249908 , -73.507788  ]
    ])

    distance_matrix = osrm_distance_matrix(
       sources,
       destinations,
       osrm_server_address="http://router.project-osrm.org",
    )  # outputs a 3 x 1 matrix

Again, this may have no use in a TSP instance, but it is there in case you need.

While the distance matrix is is *meters*, the durations come in *seconds*.


With all that said, please, have parsimony and do not send large amounts of data to this server. If you have a large input, the best approach is to setup a local server as mentioned before.

Obs.: Keep in mind that the distance matrix here is in general not symmetric, since the distance from point A to B may be different from B to A due to one-way streets or other factors. Fortunately, this is not a problem to the algorithms of this library as they can handle them.


TSPLIB
======

Finally, this module also has support for many TSPLIB-type files of ``TSP`` and ``ATSP`` format. Just enter the file and a proper distance matrix is returned.


.. code:: python

    from python_tsp.distances import tsplib_distance_matrix

    tsplib_file = "tests/tsplib_data/br17.atsp"  # replace with the path to your TSPLIB file
    distance_matrix = tsplib_distance_matrix(tsplib_file)
    # outputs a 17 x 17 array
