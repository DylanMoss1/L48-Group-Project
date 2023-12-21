from world import World
from typing import Dict, Any

grid_length_size = 100


class MainSimulator:
    """
    A class containing the ground truth simulator. 
    """

    def __init__(self) -> None:
        self.world = World(grid_length_size)

    def run(self, mutation_rates) -> Dict[str, Any]:
        return self.world.run(mutation_rates)

