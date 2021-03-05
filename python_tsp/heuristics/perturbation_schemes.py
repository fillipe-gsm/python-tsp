"""Perturbation schemes used in local search-based algorithms.
The functions here should receive a permutation list with numbers from 0 to `n`
and return a generator with all neighbors of this permutation.
The neighbors should preferably be randomized to be used in Simulated
Annealing, which samples a single neighbor at a time.
"""
from typing import Callable, Dict, Generator, List

import random


def ps1_gen(x: List[int]) -> Generator[List[int], List[int], None]:
    """PS1 perturbation scheme: Swap two adjacent terms
    This scheme has at most n - 1 swaps.
    """

    n = len(x)
    i_range = range(1, n - 1)
    for i in random.sample(i_range, len(i_range)):
        xn = x.copy()
        xn[i], xn[i + 1] = x[i + 1], xn[i]
        yield xn


def ps2_gen(x: List[int]) -> Generator[List[int], List[int], None]:
    """PS2 perturbation scheme: Swap any two elements
    This scheme has n * (n - 1) / 2 swaps.
    """

    n = len(x)
    i_range = range(1, n - 1)
    for i in random.sample(i_range, len(i_range)):
        j_range = range(i + 1, n)
        for j in random.sample(j_range, len(j_range)):
            xn = x.copy()
            xn[i], xn[j] = xn[j], xn[i]
            yield xn


def ps3_gen(x: List[int]) -> Generator[List[int], List[int], None]:
    """PS3 perturbation scheme: A single term is moved
    This scheme has n * (n - 1) swaps.
    """

    n = len(x)
    i_range = range(1, n)
    for i in random.sample(i_range, len(i_range)):
        j_range = [j for j in range(1, n) if j != i]
        for j in random.sample(j_range, len(j_range)):
            xn = x.copy()
            node = xn.pop(i)
            xn.insert(j, node)
            yield xn


# Mapping with all possible neighborhood generators in this module
neighborhood_gen: Dict[
    str, Callable[[List[int]], Generator[List[int], List[int], None]]
] = {
    "ps1": ps1_gen, "ps2": ps2_gen, "ps3": ps3_gen,
}
