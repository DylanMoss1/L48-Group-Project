import numpy as np

from world import LogItem


def day_to_temperature(day: np.array):
    return 10 + 18 * np.sin(2 * np.pi * day / 100) + (day / 100)


def logitem_to_vector(item: LogItem):
    return np.array(
        [
            item.day,
            item.num_species_alive,
            item.temperature,
            np.mean(item.traits_dict["size"]),
            np.mean(item.traits_dict["speed"]),
            np.mean(item.traits_dict["vision"]),
            np.mean(item.traits_dict["aggression"]),
            np.mean(item.traits_dict["energy"]),
        ]
    )


def load_data_from_file(x_file_path, y_file_path):
    return np.load(x_file_path), np.load(y_file_path)


def format_data_for_drift_model(x, y):
    # convert day input to temperature
    x[:, 0] = day_to_temperature(x[:, 0])
    X = [x for i in range(4)]
    Y = [y[:, 3 + i][..., None] for i in range(4)]
    return X, Y


def format_data_for_population_model(x, y):
    x[:, 0] = day_to_temperature(x[:, 0])
    X = x[:, 0:6]
    Y = y[:, 1][..., None]
    return X, Y
