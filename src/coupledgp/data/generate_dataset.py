import numpy as np

from emukit.core.initial_designs.latin_design import LatinDesign

from simulator import MainSimulator
from world import DebugInfo
from ..utils import logitems_to_vector, training_space, test_space


def generate_test_data(n_samples, save_location: str = None):
    simulator = MainSimulator()
    design = LatinDesign(test_space)
    X = design.get_samples(n_samples)
    Y = []
    for inputs in X:
        days_survived = simulator.run(
            mutation_rates={
                "size": inputs[0],
                "speed": inputs[1],
                "vision": inputs[2],
                "aggression": inputs[3],
            }
        )
        Y.append(days_survived)
    if save_location is None:
        save_location = ""
    np.save(
        f"{save_location}x-sim-{n_samples}-test",
        X,
    )
    np.save(
        f"{save_location}y-sim-{n_samples}-test",
        np.array(Y).reshape(len(Y), 1),
    )


def generate_data(n_samples: int, n_steps: int, save_location: str = None):
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
        X, n_steps
    )  # shape (n_samples * 500~ish x n_outputs) for Y (need to add in X inputs from sim to match shape)

    if save_location is None:
        save_location = ""
    np.save(
        f"{save_location}x-sim-{n_samples}-{n_steps}",
        X,
    )
    np.save(
        f"{save_location}y-sim-{n_samples}-{n_steps}",
        Y,
    )

    return X, Y


def simulate(X, n_steps):
    """X in the form (day, population, size, speed, vision, aggression, var_size, var_speed, var_vision, var_aggression, m_size, m_speed, m_vision, m_aggression)"""
    simulator = MainSimulator()
    all_inputs = []
    all_results = []
    for inputs in X:
        sim_results = logitems_to_vector(
            simulator.run_from_start_point(  # input in the form (day, population, m1, ..., m4, t1, ..., t4)
                mutation_rates={
                    "size": inputs[10],
                    "speed": inputs[11],
                    "vision": inputs[12],
                    "aggression": inputs[13],
                },
                day_start_point=inputs[0],
                population_start_point=inputs[1],
                mutation_start_point={
                    "size": (inputs[2], inputs[6]),
                    "speed": (inputs[3], inputs[7]),
                    "vision": (inputs[4], inputs[8]),
                    "aggression": (inputs[5], inputs[9]),
                },
                max_days=inputs[0] + n_steps,
            )[
                1
            ]
        )
        sim_inputs = np.delete(sim_results, 2, 1)[
            :-1, :
        ]  # inputs are previous day's output for sim (minus temperature)
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
