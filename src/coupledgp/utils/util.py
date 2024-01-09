import numpy as np

from world import LogItem


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


def format_data_for_drift_model(x, y, x_file_path=None, y_file_path=None):
    if x is None:
        x = np.load(x_file_path)
    if y is None:
        y = np.load(y_file_path)

    X = [x for i in range(4)]
    Y = [y[:, 3 + i] for i in range(4)]
    return X, Y


def format_data_for_population_model(x, y, x_file_path=None, y_file_path=None):
    if x is None:
        x = np.load(x_file_path)
    if y is None:
        y = np.load(y_file_path)

    X = x[:, 0:6]
    Y = y[:, 1]
    return X, Y
