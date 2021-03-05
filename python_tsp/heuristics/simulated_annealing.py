from typing import List, Optional, Tuple

import numpy as np

from python_tsp.heuristics.perturbation_schemes import neighborhood_gen
from python_tsp.heuristics.local_search import setup
from python_tsp.utils import compute_permutation_distance


def solve_tsp_simulated_annealing(
    distance_matrix: np.ndarray,
    x0: Optional[List[int]] = None,
    perturbation_scheme: str = "ps3",
    alpha: float = 0.9,
    verbose: bool = False,
) -> Tuple[List, float]:
    """Solve a TSP problem using a Simulated Annealing
    The approach used here is the one proposed in [1].

    Parameters
    ----------
    distance_matrix
        Distance matrix of shape (n x n) with the (i, j) entry indicating the
        distance from node i to j

    x0
        Initial permutation. If not provided, it uses a random value

    perturbation_scheme {"ps1", "ps2", ["ps3"]}
        Mechanism used to generate new solutions. Defaults to PS3. See [2] for
        a quick explanation on these schemes.

    alpha
        Reduction factor (``alpha`` < 1) used to reduce the temperature. As a
        rule of thumb, 0.99 takes longer but may return better solutions, whike
        0.9 is faster but may not be as good. A good approach is to use 0.9
        (as default) and if required run the returned solution with a local
        search.

    verbose
        `True` to display information about the process.

    Returns
    -------
    A permutation of nodes from 0 to n that produces the least total distance
    obtained (not necessarily optimal).

    The total distance the returned permutation produces.

    References
    ----------
    [1] Dréo, Johann, et al. Metaheuristics for hard optimization: methods and
    case studies. Springer Science & Business Media, 2006.

    [2] Goulart, Fillipe, et al. "Permutation-based optimization for the load
    restoration problem with improved time estimation of maneuvers."
    International Journal of Electrical Power & Energy Systems 101 (2018):
    339-355.
    """

    x, fx = setup(distance_matrix, x0)
    temp = _initial_temperature(distance_matrix, x, fx, perturbation_scheme)

    n = len(x)
    k_inner_min = 12 * n  # min inner iterations
    k_inner_max = 100 * n  # max inner iterations
    k_noimprovements = 0  # number of inner loops without improvement

    while k_noimprovements < 3:
        k_accepted = 0  # number of accepted perturbations
        for k in range(k_inner_max):
            xn = _perturbation(x, perturbation_scheme)
            fn = compute_permutation_distance(distance_matrix, xn)

            if _acceptance_rule(fx, fn, temp):
                x, fx = xn, fn
                k_accepted += 1
                k_noimprovements = 0

            if verbose:
                print((
                    f"Temperature {temp}. Current value: {fx} "
                    f"k: {k + 1}/{k_inner_max} "
                    f"k_accepted: {k_accepted}/{k_inner_min} "
                    f"k_noimprovements: {k_noimprovements} "
                ), end="\r")

            if k_accepted >= k_inner_min:
                break

        temp *= alpha  # temperature update
        k_noimprovements += k_accepted == 0
        if verbose:
            print("")  # line break

    return x, fx


def _initial_temperature(
    distance_matrix: np.ndarray,
    x: List[int],
    fx: float,
    perturbation_scheme: str = "ps3"
) -> float:
    """Compute initial temperature
    Instead of relying on problem-dependent parameters, this function estimates
    the temperature using the suggestion in [1].

    Notes
    -----
    Here are the steps followed:
        1. Generate 100 disturbances at random from T0, and evaluate the mean
        objective value differences dfx_mean = mean(fn - fx);
        2. Choose tau0 = 0.5 as assumed quality of initial solution (assuming
        a bad one), and deduce T0 from exp(-fx_mean/T0) = tau0, that is,
        T0 = -fx_mean/ln(tau0)

    References
    ----------
    [1] Dréo, Johann, et al. Metaheuristics for hard optimization: methods and
    case studies. Springer Science & Business Media, 2006.
    """

    # Step 1
    dfx_list = []
    for _ in range(100):
        xn = _perturbation(x, perturbation_scheme)
        fn = compute_permutation_distance(distance_matrix, xn)
        dfx_list.append(fn - fx)

    dfx_mean = np.abs(np.mean(dfx_list))

    # Step 2
    tau0 = 0.5
    return -dfx_mean / np.log(tau0)


def _perturbation(x: List[int], perturbation_scheme: str = "ps3"):
    """Generate a random neighbor of a current solution ``x``
    In this case, we can use the generators created in the `local_search`
    module, and pick the first solution. Since the neighborhood is randomized,
    it is the same as creating a random perturbation.
    """
    return next(neighborhood_gen[perturbation_scheme](x))


def _acceptance_rule(fx: float, fn: float, temp: float) -> bool:
    """Metropolis acceptance rule"""

    dfx = fn - fx
    return (dfx < 0) or (
        (dfx > 0) and (np.random.rand() <= np.exp(-(fn - fx) / temp))
    )
