import numpy as np
from .util import day_to_temperature, NUM_TRAITS


def coupled_to_population(x: np.ndarray, day: int, traits: np.ndarray):
    """
    Convert inputs from the coupled emulator into inputs for the population emulator.
    The coupled emulator inputs are in the form (n_samples x [input_index, population, m_size, m_speed, m_vision, m_aggression]).
    The population emulator requires inputs of the form (n_samples x [input_index, sample_index, temperature, population, size, speed, vision, aggression, size_var, speed_var, vision_var, aggression_var]).
    """
    # add temperature
    x = np.insert(x, 1, day_to_temperature(day), axis=1)
    # add traits
    x = np.insert(x, [3], traits, axis=1)
    # remove mutation rates
    x = x[:, 0:-4]
    return x


def population_to_drift(
    prev_x: np.ndarray,
    prev_y_mean: np.ndarray,
    prev_y_var: np.ndarray,
    day: int,
    n_samples: int,
    rng: np.random.Generator,
    mutation_rates: np.ndarray,
):
    """
    Convert results from the population emulator into inputs for the next step drift emulator.

    'mutation_rates' are in the form (4)

    'prev_x' is in the form (n_samples x [input_index, sample_index, temperature, population, size, speed, vision, aggression, size_var, speed_var, vision_var, aggression_var])

    'prev_y_mean' is in the form (1)
    'prev_y_var' is in the form (1)

    'next_x' is in the form ([n_samples * n_outputs] x [input_index, sample_index, temperature, population, size, speed, vision, aggression, size_var, speed_var, vision_var, aggression_var, m_size, m_speed, m_vision, m_aggression, output])
    """
    # replace temperature
    prev_x[:, 2] = day_to_temperature(day)
    # add mutation rates ([n_inputs * n_samples] x 16)
    next_x = np.hstack(
        [prev_x, np.tile(mutation_rates[np.newaxis, ...], (len(prev_x), 1))]
    )
    # generate new population from distribution (1)
    new_population = rng.normal(prev_y_mean, prev_y_var, (len(next_x),))
    # replace old population with new population
    next_x[:, 3] = new_population

    # duplicate with outputs
    next_x = np.vstack(
        [
            np.append(
                next_x,
                np.full((next_x.shape[0], 1), i),
                axis=1,
            )
            for i in range(NUM_TRAITS)
        ]
    )
    return next_x


def drift_to_population(
    prev_x: np.ndarray,
    prev_y_mean: np.ndarray,
    prev_y_var: np.ndarray,
    day: int,
    n_samples: int,
    rng: np.random.Generator,
):
    """
    Convert inputs from the drift emulator into inputs for the next step population emulator.

    'prev_x' is in the form ([n_samples * n_outputs] x [input_index, sample_index, temperature, population, size, speed, vision, aggression, size_var, speed_var, vision_var, aggression_var, m_size, m_speed, m_vision, m_aggression, output])

    'prev_y_mean' is in the form ([n_outputs] x 1)
    'prev_y_var' is in the form ([n_outputs] x 1)

    'next_x' is in the form (n_samples x [input_index, sample_index, temperature, population, size, speed, vision, aggression, size_var, speed_var, vision_var, aggression_var])
    """
    # remove output and mutation rate from inputs (n_samples x 12)
    next_x = np.vsplit(prev_x[:, :-5], NUM_TRAITS)[0]
    # replace days
    next_x[:, 2] = day_to_temperature(day)
    # generate new traits (n_samples x n_outputs)
    new_traits = rng.normal(
        prev_y_mean, prev_y_var, (prev_y_mean.shape[0], len(next_x))
    ).transpose()
    # replace traits (n_inputs x 12)
    next_x[:, 4:] = new_traits
    return next_x
