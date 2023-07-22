from typing import Tuple

import numpy as np

INF = np.iinfo(int).max


def compute_reduced_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Compute the reduced matrix and the total reduction cost.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix containing integer values.

    Returns
    -------
    Tuple[np.ndarray, int]
        A tuple containing the reduced matrix and the total reduction
        cost.

    Notes
    -----
    - The input matrix is expected to contain integer values, where INF
    (maximum integer value) represent an "infinite" value or an invalid
    entry in the matrix.
    - The reduction process involves subtracting the minimum value from
    each row and column of the matrix when the minimum value is not INF,
    effectively reducing the matrix size.
    - If a row or column contains only INF values, it is not considered
    in the reduction process.
    - The total reduction cost is the sum of the minimum values removed
    from the rows and columns during reduction.
    """
    mask = matrix != INF
    reduced_matrix = np.copy(matrix)

    min_rows = reduced_matrix.min(axis=1, keepdims=True)
    min_rows[min_rows == INF] = 0
    if np.any(min_rows != 0):
        reduced_matrix = np.where(
            mask, reduced_matrix - min_rows, reduced_matrix
        )

    min_cols = reduced_matrix.min(axis=0, keepdims=True)
    min_cols[min_cols == INF] = 0
    if np.any(min_cols != 0):
        reduced_matrix = np.where(
            mask, reduced_matrix - min_cols, reduced_matrix
        )

    return reduced_matrix, min_rows.sum() + min_cols.sum()
