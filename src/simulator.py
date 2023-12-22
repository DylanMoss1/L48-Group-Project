from world import LogItem, World, DebugInfo
from typing import Dict, Any, List
from abc import ABC


class Simulator(ABC):
    """
    An abstract class representing a simulator, parameterised on grid_length_size. 
    """

    def __init__(self, grid_length_size) -> None:
        """
        Initialise the Simulator object. 

        Parameters
        ----------
        grid_length_size : int 
            Length of each size of the simulation grid.
        """
        self.world = World(grid_length_size)

    def run(self, mutation_rates, debug_info=DebugInfo()) -> (int, List[LogItem]):
        """
        Run the simulation with given mutation rates until the species goes extinct.

        Parameters
        ----------
        mutation_rates : dict(string, int)
            Contains keys: size, speed, vision, aggression. 
            With corresponding values representing the mutation rates for each trait
        debug_info : DebugInfo 
            Determines how much information should be printed to the console during the program's execution (default is no debug info)

        Returns
        -------
        days_survived : int 
            The number of days the species has survived until extinction 
        log : list(LogItem)
            A list of log item entries (important values for emulation training: see world.LogItem) made throughout the simulation's execution
        """
        days_survived, log = self.world.run(mutation_rates, debug_info)
        return days_survived, log


class MainSimulator(Simulator):
    """
    A class containing the ground truth simulator. This has a 100 x 100 grid. 
    """

    def __init__(self):
        super().__init__(100)


class SmallSimulator(Simulator):
    """
    A class containing a smaller simulator than the ground truth model. This has a 20 x 20 grid.
    """

    def __init__(self):
        super().__init__(20)


class TinySimulator(Simulator):
    """
    A class containing a *even* smaller simulator than the ground truth model. This has a 10 x 10 grid.
    """

    def __init__(self):
        super().__init__(10)
