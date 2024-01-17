import numpy as np
import random
from typing import List

from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.core.loop.user_function import UserFunction, UserFunctionResult
from GPy.util.multioutput import build_XY

from simulator import MainSimulator
from ..utils import *


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
        f"{save_location}x-{n_samples}-simfor-{n_steps}-{NUM_TRAITS}-traits",
        X,
    )
    np.save(
        f"{save_location}y-{n_samples}-simfor-{n_steps}-{NUM_TRAITS}-traits",
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


def simulate_coupled(X, n_steps):
    simulator = MainSimulator()
    all_inputs = []
    all_results = []
    survival_days = []
    for inputs in X:
        sim_results = logitems_to_vector(
            simulator.run_from_start_point(  # input in the form (day, population, m1, ..., m4, t1, ..., t4)
                mutation_rates={
                    "size": inputs[1],
                    "speed": inputs[2],
                    "vision": inputs[3],
                    "aggression": inputs[4],
                },
                day_start_point=0,
                population_start_point=inputs[0],
                mutation_start_point={
                    "size": (random.random(), 0),
                    "speed": (random.random(), 0),
                    "vision": (random.random(), 0),
                    "aggression": (random.random(), 0),
                },
                max_days=n_steps,
            )[
                1
            ]
        )
        survival_days.append(sim_results[-1, 0])
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
    return np.vstack(all_inputs), np.vstack(all_results), survival_days


def simulate_drift(X, n_steps):
    X[:, 0] = temperature_to_day(X[:, 0])
    new_inputs, outputs = simulate(X[:, -1], n_steps)
    x_list, y_list = format_data_for_drift_model(new_inputs, outputs)
    x_drift, y_drift, _ = build_XY(x_list, y_list)
    return x_drift, y_drift


def simulate_population(X, n_steps):
    X[:, 0] = temperature_to_day(X[:, 0])
    X = np.hstack([X, np.zeros(X.shape[0], 4)])
    new_inputs, outputs = simulate(X, n_steps)
    x_pop, y_pop = format_data_for_population_model(new_inputs, outputs)
    return x_pop, y_pop


class SimulateCoupled(UserFunction):
    def __init__(self, n_steps: int = None):
        self.n_steps = n_steps

    def evaluate(self, X: np.array) -> List[UserFunctionResult]:
        old_input = X
        new_inputs, outputs, survival_days = simulate_coupled(X, self.n_steps)
        return [
            CustomUserFunctionResult(
                old_input,
                survival_days,
                data_input=new_inputs,
                data_output=outputs,
            )
        ]


class SimulateDrift(UserFunction):
    def __init__(self, n_steps: int = 1):
        self.n_steps = n_steps

    def evaluate(self, X: np.ndarray) -> List[UserFunctionResult]:
        new_inputs, outputs = simulate_drift(X, self.n_steps)
        return [
            UserFunctionResult(new_inputs[i], outputs[i])
            for i in range(new_inputs.shape[0])
        ]


class SimulatePopulation(UserFunction):
    def __init__(self, n_steps: int = 1):
        self.n_steps = n_steps

    def evaluate(self, X: np.ndarray) -> List[UserFunctionResult]:
        new_inputs, outputs = simulate_population(X, self.n_steps)
        return [
            UserFunctionResult(new_inputs[i], outputs[i])
            for i in range(new_inputs.shape[0])
        ]
