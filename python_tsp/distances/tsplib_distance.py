import numpy as np
import tsplib95


def tsplib_distance_matrix(tsplib_file: str) -> np.ndarray:
    """Distance matrix from a TSPLIB file

    Parameters
    ----------
    tsplib_file
        A string with the complete path of the TSPLIB file (or just its name if
        it is the in current path)

    Returns
    -------
    distance_matrix
        A ND-array with the equivalent distance matrix of the input file

    Notes
    -----
    This function can handle any file supported by the `tsplib95` lib.
    """

    tsp_problem = tsplib95.load(tsplib_file)
    distance_matrix_flattened = np.array(
        [tsp_problem.get_weight(*edge) for edge in tsp_problem.get_edges()]
    )
    distance_matrix = np.reshape(
        distance_matrix_flattened,
        (tsp_problem.dimension, tsp_problem.dimension),
    )

    # Some problems with EXPLICIT matrix have a large number in the distance
    # from a node to itself, which makes no sense for our problems. Thus,
    # always ensure a diagonal filled with zeros
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix
