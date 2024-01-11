import yaml
from pathlib import Path

# Parse simulation_constants.yml as a python dictionary
simulation_constants = yaml.safe_load(
    Path('config/simulation_constants.yml').read_text())

# Assign dictionary contents to python constant variables

# -- SEE ./config/simulation_constants.yml FOR EXPLAINATIONS OF CONSTANTS -- 

NUM_INITIAL_SPECIES_FRACTION = simulation_constants["NUM_INITIAL_SPECIES_FRACTION"]
NUM_ACTIONS_PER_DAY = simulation_constants["NUM_ACTIONS_PER_DAY"]

FOOD_PROBABILITY_SCALAR = simulation_constants["FOOD_PROBABILITY_SCALAR"]
FOOD_PROBABILITY_STD = simulation_constants["FOOD_PROBABILITY_STD"]
OPTIMAL_TEMPERATURE = simulation_constants["OPTIMAL_TEMPERATURE"]

MAXIMUM_SPEED = simulation_constants["MAXIMUM_SPEED"]


INITIAL_ENERGY = simulation_constants["INITIAL_ENERGY"]
ENERGY_LOSS = simulation_constants["ENERGY_LOSS"]

# INITIAL_SIZE = simulation_constants["INITIAL_SIZE"]
# INITIAL_SPEED = simulation_constants["INITIAL_SPEED"]
# INITIAL_VISION = simulation_constants["INITIAL_VISION"]
# INITIAL_AGGRESSION = simulation_constants["INITIAL_AGGRESSION"]

FOOD_VALUE = simulation_constants["FOOD_VALUE"]
DAMAGE_VALUE = simulation_constants["DAMAGE_VALUE"]

REPRODUCTION_THRESHOLD = simulation_constants["REPRODUCTION_THRESHOLD"]
MAXIMUM_VISION = simulation_constants["MAXIMUM_VISION"]
MAXIMUM_AGE = simulation_constants["MAXIMUM_AGE"]