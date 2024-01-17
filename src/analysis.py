from simulator import TinySimulator, MainSimulator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import json

simulator = TinySimulator()

no_mutation_rates = {
    "size": 0,
    "speed": 0,
    "vision": 0,
    "aggression": 0,
}

low_mutation_rates = {
    "size": 0.01,
    "speed": 0.01,
    "vision": 0.01,
    "aggression": 0.01,
}

high_mutation_rates = {
    "size": 0.1,
    "speed": 0.1,
    "vision": 0.1,
    "aggression": 0.1,
}


def get_populations_from_log(log): 
  populations = np.array([]) 

  for log_item in log: 
    populations = np.append(populations, log_item.num_species_alive)
  
  return populations

def arrays_from_log_list(log_list):
  populations = [get_populations_from_log(log) for log in log_list]
  max_length = max(map(len, populations))
  population_arr = np.zeros((len(populations), max_length), dtype=int)

  for i, row in enumerate(populations):
      population_arr[i, :len(row)] = row

  x = np.arange(1, max_length + 1)

  y = population_arr
  y_mean = np.mean(y, axis=0)
  y_std = np.std(y, axis=0)
  y_std_above = y_mean + y_std 
  y_std_below = np.clip(y_mean - y_std, a_min=0, a_max=None)

  return x, y_mean, y_std_above, y_std_below, populations

mutation_rates_list = [no_mutation_rates, low_mutation_rates,
                   high_mutation_rates]

mutation_rates_labels = ["No Mutation Rate", "Low Mutation Rate", "High Mutation Rate"]

num_iters = 30

# data = []

for index, mutation_rates in enumerate(mutation_rates_list):
    print(mutation_rates)
    # log_list = []
    for i in range(num_iters):
        print(i)
        _, log = simulator.run(mutation_rates)
        populations = get_populations_from_log(log)
        np.savetxt(f"analysis_results/population/data-{mutation_rates_labels[index]}-{i}.txt", populations, fmt='%d')


day: int
num_species_alive: int
temperature: float
probability_of_food: float
traits_dict: Dict[str, List[float]]


        # x, y_mean, y_std_above, y_std_below, populations = arrays_from_log_list(log)
        # np.savetxt("data.txt", x, delimiter=',', fmt='%d')
        # np.savetxt("data.txt", y_mean, delimiter=',', fmt='%d')
        # np.savetxt("data.txt", y_std_above, delimiter=',', fmt='%d')
        # np.savetxt("data.txt", y_std_below, delimiter=',', fmt='%d')
        # np.savetxt("data.txt", populations, delimiter=',', fmt='%d')
    



        # log_list.append(log)
    # data.append(log_list)


# def convert_to_list_data_form(arrays):
#     x, y_mean, y_std_above, y_std_below, populations = arrays
#     return list(x), list(y_mean), list(y_std_above), list(y_std_below), list(populations)


# def convert(obj):
#     if isinstance(obj, np.int64):
#         return int(obj)
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     else:
#         raise TypeError("Type not serializable")


# stored_data = [convert_to_list_data_form(
#     arrays_from_log_list(log_list)) for log_list in data]

# file_path = "data.json"

# with open(file_path, 'w') as json_file:
#     json.dump(stored_data, json_file, default=convert)