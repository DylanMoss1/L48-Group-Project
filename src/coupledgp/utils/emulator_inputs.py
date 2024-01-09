import numpy as np


def coupled_to_population(x: np.ndarray, day: int, traits: np.ndarray):
    """
    Convert inputs from the coupled emulator into inputs for the population emulator.
    The coupled emulator inputs are in the form (n_samples x [population, m_size, m_speed, m_vision, m_aggression]).
    The population emulator requires inputs of the form (n_samples x [day, population, size, speed, vision, aggression]).
    """
    # add days
    x = np.insert(x, 0, day, axis=1)
    # add traits
    x = np.insert(x, [2], traits)
    # remove mutation rates
    return x[:, 0:6]


def population_to_population(
    prev_x: np.ndarray,
    prev_y: np.ndarray,
    day: int,
    mututation_rates: np.ndarray,
):
    """
    Convert results from the population emulator into inputs for the next step population emulator.
    The population emulator outputs are in the form (n_samples x 1) (population).
    The population emulator requires inputs of the form (n_samples x [day, population, size, speed, vision, aggression]).
    """
    # replace days
    prev_x[:, 0] = day
    # replace old population with new population
    prev_x[:, 1] = prev_y
    return prev_x


def population_to_drift(
    prev_x: np.ndarray,
    prev_y: np.ndarray,
    day: int,
    mutation_rates: np.ndarray,
):
    """
    Convert results from the population emulator into inputs for the next step drift emulator.
    The population emulator outputs are in the form (n_samples x 1) (population).
    The population emulator inputs are in the form (n_samples x [day, population, size, speed, vision, aggression]).
    The drift emulator requires inputs of the form (n_samples x [day, population, size, speed, vision, aggression, m_size, m_speed, m_vision, m_aggression, output])
    """
    # replace days
    prev_x[:, 0] = day
    # replace old population with new population
    prev_x[:, 1] = prev_y
    # add mutation rates
    prev_x = np.hstack([prev_x, mutation_rates])
    # duplicate with outputs
    prev_x = np.vstack(
        [np.insert(prev_x, -1, i) for i in range(mutation_rates.shape[1])]
    )
    return prev_x


def drift_to_drift(
    prev_x: np.ndarray,
    prev_y: np.ndarray,
    day: int,
    mututation_rates: np.ndarray,
):
    """
    Convert results from the drift emulator into inputs for the next step drift emulator.
    The drift emulator outputs are in the form ([n_samples x 4] x 1) (trait value) (i.e., stacked arrays of (n_samples x 1) for each trait)
    The drift emulator requires inputs of the form (n_samples x [day, population, size, speed, vision, aggression, m_size, m_speed, m_vision, m_aggression, output])
    """
    # traits in a single np.ndarray of (n_samples x 4)
    traits = np.hstack(np.split(prev_y, 4, axis=0))
    traits = np.tile(traits, (4, 1))
    # replace days
    prev_x[:, 0] = day
    # replace traits
    prev_x[:, 2:6] = traits
    return prev_x


def drift_to_population(
    prev_x: np.ndarray,
    prev_y: np.ndarray,
    day: int,
    mututation_rates: np.ndarray,
):
    """
    Convert inputs from the drift emulator into inputs for the next step population emulator.
    The drift emulator outputs are in the form ([n_samples x 4] x 1) (trait value) (i.e., stacked arrays of (n_samples x 1) for each trait)
    The drift emulator inputs are in the form (n_samples x [day, population, size, speed, vision, aggression, m_size, m_speed, m_vision, m_aggression, output])
    The population emulator requires inputs of the form (n_samples x [day, population, size, speed, vision, aggression]).
    """
    traits = np.hstack(np.split(prev_y, 4, axis=0))
    prev_input = np.split(prev_x, 4, axis=0)[0]
    # replace days
    prev_input[:, 0] = day
    # replace traits
    prev_input[:, 2:6] = traits
    # remove mutation rates
    return prev_input[:, 0:6]
