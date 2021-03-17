from timeit import default_timer
from pathlib import Path

from python_tsp.heuristics import (
    solve_tsp_local_search, solve_tsp_simulated_annealing
)
from python_tsp.distances import tsplib_distance_matrix


# Files ignored for being too large or having weird distance matrices
IGNORED_FILES = (
    "att48.tsp", "d18512.tsp", "att532.tsp", "pla33810.tsp", "pla85900.tsp",
)

symmetric_instances = Path("experiments/tsp/").rglob("*.tsp")
asymmetric_instances = Path("experiments/atsp/").rglob("*.atsp")

solvers = (solve_tsp_simulated_annealing, solve_tsp_local_search)
perturbation_schemes = ["ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "two_opt"]
num_replications = 1

if __name__ == "__main__":
    symmetric_data = []

    for instance_file in symmetric_instances:
        if instance_file.name in IGNORED_FILES:
            continue
        distance_matrix = tsplib_distance_matrix(instance_file)
        for perturbation_scheme in perturbation_schemes:
            for solver in solvers:
                for i in range(num_replications):
                    print(
                        f"Solver {solver.__name__} "
                        f"on instance {instance_file.name} "
                        f"scheme {perturbation_scheme}"
                    )
                    tic = default_timer()
                    _, distance = solver(
                        distance_matrix,
                        perturbation_scheme=perturbation_scheme,
                        max_processing_time=5 * 60,  # max 5 minutes
                    )
                    toc = default_timer()

                    symmetric_data.append(
                        {
                            "instance": instance_file.name,
                            "solver": solver.__name__,
                            "perturbation_scheme": perturbation_scheme,
                            "replication": i,
                            "distance": distance,
                            "processing_time": toc - tic,
                        }
                    )

    def write_line(data_row):
        return ",".join(str(value) for value in data_row.values()) + "\n"

    with open("symmetric_results.csv", "w") as f:
        header = ",".join(key for key in symmetric_data[0].keys()) + "\n"
        lines = [header] + [
            write_line(data_row) for data_row in symmetric_data
        ]
        f.writelines(lines)

# IGNORED_FILES = (
    # "att48.tsp", "d18512.tsp", "att532.tsp", "pla33810.tsp", "pla85900.tsp",
# )
# from python_tsp.distances import tsplib_distance_matrix
# from pathlib import Path
# symmetric_instances = Path("experiments/tsp/").rglob("*.tsp")

# for instance_file in symmetric_instances:
    # if instance_file.name in IGNORED_FILES:
        # print(f"Ignoring {instance_file.name}")
        # continue
    # print(instance_file)
    # distance_matrix = tsplib_distance_matrix(instance_file)
    # if distance_matrix.shape[0] == 1:
        # print(instance_file)
        # break
