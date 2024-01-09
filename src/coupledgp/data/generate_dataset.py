import numpy as np

from emukit.core import ParameterSpace, DiscreteParameter, ContinuousParameter
from emukit.core.initial_designs.latin_design import LatinDesign

from src.simulator import MainSimulator
from ..utils import logitem_to_vector


def generate_data(n_samples: int, save_location: str = None):
    """
    Generates training data based on single-timestep results

    Args:
        n_samples (int): number of samples to generate
        save_location (str, optional): folder to save results in. Defaults to None.
    """
    parameter_space = ParameterSpace(
        [
            DiscreteParameter(
                "day", range(1, 366)
            ),  # random start day within a year
            DiscreteParameter(
                "population", range(1000, 3000)
            ),  # 100 x 100 grid (10 - 30% populated)
            ContinuousParameter(
                "size", 0.5, 2.5
            ),  # need to tweak these limits possibly
            ContinuousParameter("speed", 0.5, 2.5),
            ContinuousParameter("vision", 0.5, 2.5),
            ContinuousParameter("aggression", 0.5, 2.5),
            ContinuousParameter("m_size", 0, 5),
            ContinuousParameter("m_speed", 0, 5),
            ContinuousParameter("m_vision", 0, 5),
            ContinuousParameter("m_aggression", 0, 5),
        ]
    )
    design = LatinDesign(parameter_space)
    X = design.get_samples(n_samples)  # shape (n_samples x num_inputs)
    Y = simulate(X)  # shape (n_samples x n_outputs)

    if save_location is None:
        save_location = ""
    np.save(
        f"{save_location}x-{n_samples}",
        X,
    )
    np.save(
        f"{save_location}y-{n_samples}",
        Y,
    )

    return X, Y


def simulate(X):
    simulator = MainSimulator()
    results = [
        logitem_to_vector(
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
                    "size": (inputs[6], 1),
                    "speed": (inputs[7], 1),
                    "vision": (inputs[8], 1),
                    "aggression": (inputs[9], 1),
                },
                max_days=inputs[0] + 1,
            )[
                1
            ][
                0
            ]
        )
        for inputs in X
    ]  # results in the form [(days_survived, [LogItem]), ...] -> [LogItem, ...] -> [np.array (n_inputs), ...]
    return np.vstack(results)
