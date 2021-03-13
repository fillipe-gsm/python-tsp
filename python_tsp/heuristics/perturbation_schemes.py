"""Perturbation schemes used in local search-based algorithms.
The functions here should receive a permutation list with numbers from 0 to `n`
and return a generator with all neighbors of this permutation.
The neighbors should preferably be randomized to be used in Simulated
Annealing, which samples a single neighbor at a time.

References
----------
.. [1] Goulart, Fillipe, et al. "Permutation-based optimization for the load
    restoration problem with improved time estimation of maneuvers."
    International Journal of Electrical Power & Energy Systems 101 (2018):
    339-355.
.. [2] 2-opt: https://en.wikipedia.org/wiki/2-opt
"""
from random import sample
from typing import Callable, Dict, Generator, List


def ps1_gen(x: List[int]) -> Generator[List[int], List[int], None]:
    """PS1 perturbation scheme: Swap two adjacent terms [1]
    This scheme has at most n - 1 swaps.
    """

    n = len(x)
    i_range = range(1, n - 1)
    for i in sample(i_range, len(i_range)):
        xn = x.copy()
        xn[i], xn[i + 1] = xn[i + 1], xn[i]
        yield xn


def ps2_gen(x: List[int]) -> Generator[List[int], List[int], None]:
    """PS2 perturbation scheme: Swap any two elements [1]
    This scheme has n * (n - 1) / 2 swaps.
    """

    n = len(x)
    i_range = range(1, n - 1)
    for i in sample(i_range, len(i_range)):
        j_range = range(i + 1, n)
        for j in sample(j_range, len(j_range)):
            xn = x.copy()
            xn[i], xn[j] = xn[j], xn[i]
            yield xn


def ps3_gen(x: List[int]) -> Generator[List[int], List[int], None]:
    """PS3 perturbation scheme: A single term is moved [1]
    This scheme has n * (n - 1) swaps.
    """

    n = len(x)
    i_range = range(1, n)
    for i in sample(i_range, len(i_range)):
        j_range = [j for j in range(1, n) if j != i]
        for j in sample(j_range, len(j_range)):
            xn = x.copy()
            node = xn.pop(i)
            xn.insert(j, node)
            yield xn


def ps4_gen(x: List[int]) -> Generator[List[int], List[int], None]:
    """PS4 perturbation scheme: A subsequence is moved [1]"""

    n = len(x)
    i_range = range(1, n)
    for i in sample(i_range, len(i_range)):
        j_range = range(i + 1, n + 1)
        for j in sample(j_range, len(j_range)):
            k_range = [k for k in range(1, n + 1) if k not in range(i, j + 1)]
            for k in sample(k_range, len(k_range)):
                xn = x.copy()
                if k < i:
                    xn = xn[:k] + xn[i:j] + xn[k:i] + xn[j:]
                else:
                    xn = xn[:i] + xn[j:k] + xn[i:j] + xn[k:]
                yield xn


def ps5_gen(x: List[int]) -> Generator[List[int], List[int], None]:
    """PS5 perturbation scheme: A subsequence is reversed [1]"""

    n = len(x)
    i_range = range(1, n)
    for i in sample(i_range, len(i_range)):
        j_range = range(i + 2, n + 1)
        for j in sample(j_range, len(j_range)):
            xn = x.copy()
            xn = xn[:i] + list(reversed(xn[i:j])) + xn[j:]
            yield xn


def ps6_gen(x: List[int]) -> Generator[List[int], List[int], None]:
    """PS6 perturbation scheme: A subsequence is reversed and moved [1]"""

    n = len(x)
    i_range = range(1, n)
    for i in sample(i_range, len(i_range)):
        j_range = range(i + 1, n + 1)
        for j in sample(j_range, len(j_range)):
            k_range = [k for k in range(1, n + 1) if k not in range(i, j + 1)]
            for k in sample(k_range, len(k_range)):
                xn = x.copy()
                if k < i:
                    xn = xn[:k] + list(reversed(xn[i:j])) + xn[k:i] + xn[j:]
                else:
                    xn = xn[:i] + xn[j:k] + list(reversed(xn[i:j])) + xn[k:]
                yield xn


def two_opt_gen(x: List[int]) -> Generator[List[int], List[int], None]:
    """2-opt perturbation scheme [2]"""
    n = len(x)
    i_range = range(2, n)
    for i in sample(i_range, len(i_range)):
        j_range = range(i + 1, n + 1)
        for j in sample(j_range, len(j_range)):
            xn = x.copy()
            xn = xn[:i - 1] + list(reversed(xn[i - 1:j])) + xn[j:]
            yield xn


# Mapping with all possible neighborhood generators in this module
neighborhood_gen: Dict[
    str, Callable[[List[int]], Generator[List[int], List[int], None]]
] = {
    "ps1": ps1_gen,
    "ps2": ps2_gen,
    "ps3": ps3_gen,
    "ps4": ps4_gen,
    "ps5": ps5_gen,
    "ps6": ps6_gen,
    "two_opt": two_opt_gen,
}
