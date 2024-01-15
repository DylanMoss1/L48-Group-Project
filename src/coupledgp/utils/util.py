from typing import List
import numpy as np

from world import LogItem

NUM_TRAITS = 4


def day_to_temperature(day: np.array):
    return 10 + 18 * np.sin(2 * np.pi * day / 100) + (day / 100)


def logitems_to_vector(items: List[LogItem]):
    result = np.vstack(
        [
            np.array(
                [
                    item.day,
                    item.num_species_alive,
                    item.temperature,
                    np.mean(item.traits_dict["size"]),
                    np.mean(item.traits_dict["speed"]),
                    np.mean(item.traits_dict["vision"]),
                    np.mean(item.traits_dict["aggression"]),
                    np.var(item.traits_dict["size"]),
                    np.var(item.traits_dict["speed"]),
                    np.var(item.traits_dict["vision"]),
                    np.var(item.traits_dict["aggression"]),
                ]
            )
            for item in items
        ]
    )
    return result


def load_data_from_file(x_file_path, y_file_path):
    return np.load(x_file_path), np.load(y_file_path)


def format_data_for_drift_model(x, y):
    """X is in the format of the array above minus temperature + mutation rates at the end, Y is in the format of the array above"""
    # convert day input to temperature
    x[:, 0] = day_to_temperature(x[:, 0])
    X = [x for _ in range(NUM_TRAITS)]
    Y = [y[:, 3 + i][..., None] for i in range(NUM_TRAITS)]
    return X, Y


def format_data_for_population_model(x, y):
    x[:, 0] = day_to_temperature(x[:, 0])
    X = x[:, : 2 + NUM_TRAITS]
    Y = y[:, 1][..., None]
    return X, Y
