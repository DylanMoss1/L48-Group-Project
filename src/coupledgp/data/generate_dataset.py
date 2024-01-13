import numpy as np

from emukit.core.initial_designs.latin_design import LatinDesign

from simulator import MainSimulator
from world import DebugInfo
from ..utils import logitems_to_vector, training_space


def generate_data(n_samples: int, save_location: str = None):
    """
    Generates training data based on single-timestep results

    Args:
        n_samples (int): number of simulations to run to generate
        save_location (str, optional): folder to save results in. Defaults to None.
    """
    parameter_space = training_space
    design = LatinDesign(parameter_space)
    X = design.get_samples(n_samples)  # shape (n_samples x num_inputs)
    X, Y = simulate(
        X
    )  # shape (n_samples * 500~ish x n_outputs) for Y (need to add in X inputs from sim to match shape)

    if save_location is None:
        save_location = ""
    np.save(
        f"{save_location}x-sim-{n_samples}",
        X,
    )
    np.save(
        f"{save_location}y-sim-{n_samples}",
        Y,
    )

    return X, Y


def simulate(X):
    simulator = MainSimulator()
    all_inputs = []
    all_results = []
    for inputs in X:
        sim_results = logitems_to_vector(
            simulator.run_from_start_point(  # input in the form (day, population, m1, ..., m4, t1, ..., t4)
                mutation_rates={
                    "size": inputs[2],
                    "speed": inputs[3],
                    "vision": inputs[4],
                    "aggression": inputs[5],
                },
                day_start_point=inputs[0],
                population_start_point=inputs[1],
                mutation_start_point={
                    "size": (inputs[6], 0),
                    "speed": (inputs[7], 0),
                    "vision": (inputs[8], 0),
                    "aggression": (inputs[9], 0),
                },
                debug_info=DebugInfo(
                    period=10,
                    should_display_day=True,
                    should_display_population=True,
                ),
                max_days=inputs[0] + 500,
            )[
                1
            ]
        )
        sim_inputs = sim_results[
            :-1, [0, 1, 3, 4, 5, 6]
        ]  # inputs are previous day's output for sim
        sim_inputs = np.insert(
            sim_inputs,
            [2],
            [inputs[2], inputs[3], inputs[4], inputs[5]],
            axis=1,
        )
        sim_inputs = np.vstack([inputs, sim_inputs])
        all_inputs.append(sim_inputs)
        all_results.append(sim_results)
    # results in the form [(days_survived, [LogItem]), ...] -> [LogItem, ...] -> [np.array (n_days, n_outputs), ...]
    return np.vstack(all_inputs), np.vstack(all_results)
