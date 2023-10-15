from typing import List

import numpy as np
from geopy import distance

from api.datamodel import City
from python_tsp.exact import solve_tsp_dynamic_programming


def _compute_distance_matrix(cities: List[City]):
    distance_matrix = np.zeros((len(cities), len(cities)))
    for i, city1 in enumerate(cities):
        coords_1 = (city1.latitude, city1.longitude)
        for j, city2 in enumerate(cities):
            if j > i:
                coords_2 = (city2.latitude, city2.longitude)

                dist = distance.geodesic(coords_1, coords_2).km
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

    return distance_matrix


def compute_optimal_tour(cities: List[City]):
    distance_matrix = _compute_distance_matrix(cities)
    permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
    optimal_tour = [cities[i].name for i in permutation]
    return (optimal_tour, distance)
