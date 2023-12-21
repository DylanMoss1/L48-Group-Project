import yaml
from pathlib import Path


simulation_constants = yaml.safe_load(
    Path('src/config/simulation_constants.yml').read_text())

NUM_INITIAL_SPECIES = simulation_constants["NUM_INITIAL_SPECIES"]
