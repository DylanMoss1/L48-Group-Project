import numpy as np
from .util import day_to_temperature


def coupled_to_population(x: np.ndarray, day: int, traits: np.ndarray):
    """
    Convert inputs from the coupled emulator into inputs for the population emulator.
    The coupled emulator inputs are in the form (n_samples x [index, population, m_size, m_speed, m_vision, m_aggression]).
    The population emulator requires inputs of the form (n_samples x [index, temperature, population, size, speed, vision, aggression]).
    """
    # add days
    x = np.insert(x, 1, day, axis=1)
    # add traits
    x = np.insert(x, [3], traits, axis=1)
    # remove mutation rates
    x = x[:, 0:7]
    return x


def population_to_population(
    prev_x: np.ndarray,
    prev_y: np.ndarray,
    day: int,
    mutation_rates: np.ndarray,
):
    """
    Convert results from the population emulator into formatted population (n_samples x 1) and inputs for the next step population emulator.
    The population emulator outputs are in the form (n_samples x 1) (population).
    The population emulator requires inputs of the form (n_samples x [index, temperature, population, size, speed, vision, aggression]).
    """
    # replace days
    prev_x[:, 1] = day_to_temperature(day)
    # replace old population with new population
    prev_x[:, [2]] = prev_y
    return prev_x, mutation_rates, prev_x[:, [0, 2]]


def population_to_drift(
    prev_x: np.ndarray,
    prev_y: np.ndarray,
    day: int,
    mutation_rates: np.ndarray,
):
    """
    Convert results from the population emulator into formatted population (n_samples x 1) and inputs for the next step drift emulator.
    The population emulator outputs are in the form (n_samples x 1) (population).
    The population emulator inputs are in the form (n_samples x [index, temperature, population, size, speed, vision, aggression]).
    The drift emulator requires inputs of the form ([n_samples x 4] x [index, temperature, population, size, speed, vision, aggression, m_size, m_speed, m_vision, m_aggression, output])
    """
    # replace days
    prev_x[:, 1] = day_to_temperature(day)
    # replace old population with new population
    prev_x[:, [2]] = prev_y
    # add mutation rates
    prev_x = np.hstack([prev_x, mutation_rates])
    # duplicate with outputs
    prev_x = np.vstack(
        [
            np.append(prev_x, np.full((prev_x.shape[0], 1), i), axis=1)
            for i in range(mutation_rates.shape[1])
        ]
    )
    return prev_x, prev_x[:, 7:11], np.split(prev_x, 4, axis=0)[0][:, [0, 2]]


def drift_to_drift(
    prev_x: np.ndarray,
    prev_y: np.ndarray,
    day: int,
    mutation_rates: np.ndarray,
):
    """
    Convert results from the drift emulator into formatted outputs and inputs for the next step drift emulator.
    The drift emulator outputs are in the form ([n_samples x 4] x 1) (trait value) (i.e., stacked arrays of (n_samples x 1) for each trait)
    The drift emulator requires inputs of the form ([n_samples x 4] x [index, temperature, population, size, speed, vision, aggression, m_size, m_speed, m_vision, m_aggression, output])
    """
    # traits in a single np.ndarray of (n_samples x 4)
    traits = np.hstack(np.split(prev_y, 4, axis=0))
    traits = np.tile(traits, (4, 1))
    # replace days
    prev_x[:, 1] = day_to_temperature(day)
    # replace traits
    prev_x[:, 3:7] = traits
    return prev_x, mutation_rates, prev_x[:, [0, 2]]


def drift_to_population(
    prev_x: np.ndarray,
    prev_y: np.ndarray,
    day: int,
    mutation_rates: np.ndarray,
):
    """
    Convert inputs from the drift emulator into formatted outputs and inputs for the next step population emulator.
    The drift emulator outputs are in the form ([n_samples x 4] x 1) (trait value) (i.e., stacked arrays of (n_samples x 1) for each trait)
    The drift emulator inputs are in the form ([n_samples x 4] x [index, temperature, population, size, speed, vision, aggression, m_size, m_speed, m_vision, m_aggression, output])
    The population emulator requires inputs of the form (n_samples x [index, temperature, population, size, speed, vision, aggression]).
    """
    traits = np.hstack(np.split(prev_y, 4, axis=0))
    prev_x = np.split(prev_x, 4, axis=0)[0]
    # replace days
    prev_x[:, 1] = day_to_temperature(day)
    # replace traits
    prev_x[:, 3:7] = traits
    # remove mutation rates
    mutation_rates = prev_x[:, 7:11]
    prev_x = prev_x[:, 0:7]
    return prev_x, mutation_rates, prev_x[:, [0, 2]]
