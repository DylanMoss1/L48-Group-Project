import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from simulator import MainSimulator


def run_simulator(
    nruns: int, save_results: bool = True, save_location: str = None
) -> pd.DataFrame:
    """Runs the high-fidelity simulator several times with the same settings, and records the results.

    Args:
        nruns (int): Number of repeating runs for the simulator
        save_results (bool, optional): Save the results to a .csv file. Defaults to True.
        save_location (str, optional): Save file location. Defaults to None.
    """
    simulator = MainSimulator()
    mutation_rates = {"size": 1, "speed": 1, "vision": 1, "aggression": 1}

    data = []
    column_names = []

    for i in range(nruns):
        _, log = simulator.run(mutation_rates, max_days=500)
        last_result = log[-1]
        for k, v in vars(last_result).items():
            results = []
            match v:
                case dict():
                    for trait_name, trait_list in v.items():
                        results.append(np.mean(trait_list))
                        results.append(np.var(trait_list))

                        if i == 0:
                            column_names.append(trait_name)
                case _:
                    results.append(v)

                    if i == 0:
                        column_names.append(k)
        data.append(results)

    matrix = pd.DataFrame(np.array(data), columns=column_names)

    if save_results:
        if save_location is None:
            save_location = f"./tests/{mutation_rates['size']}-{mutation_rates['speed']}-{mutation_rates['vision']}-{mutation_rates['aggression']}-{nruns}.csv"
        matrix.to_csv(save_location)

    return matrix


def graph_histogram(
    results: pd.DataFrame, save_results: bool = True, save_location: str = None
):
    """Graphs a histogram of results from simulator runs."""
    log_names = results.columns[1:]
    fig, axes = plt.subplots(
        math.ceil(len(log_names) / 4), 4, figsize=(20, 20)
    )
    for i, name in enumerate(log_names):
        sns.histplot(results[name], ax=axes[i // 4][i % 4])

    if save_results:
        if save_location is None:
            save_location = "./tests/histogram.png"
        fig.savefig(save_location)


results = run_simulator(5)
# results = pd.read_csv("./tests/1-1-1-1-5.csv")
graph_histogram(results)
