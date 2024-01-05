from world import LogItem, World, DebugInfo
from typing import Dict, Any, List
from abc import ABC


class Simulator(ABC):
    """
    An abstract class representing a simulator, parameterised on grid_length_size. 

    Attributes 
    ----------
    world : World 
        The simulated world
    grid_length_size : int 
        Length of each size of the simulation grid
    """

    def __init__(self, grid_length_size) -> None:
        """
        Initialise the Simulator object. 

        Parameters
        ----------
        grid_length_size : int 
            Length of each size of the simulation grid
        """
        self.world = World(grid_length_size)
        self.grid_length_size = grid_length_size

    def run(self, mutation_rates, debug_info=DebugInfo(), max_days=None) -> (int, List[LogItem]):
        """
        Run the simulation with given mutation rates until the species goes extinct.

        Parameters
        ----------
        mutation_rates : dict(string, int)
            Contains keys: size, speed, vision, aggression. 
            With corresponding values representing the mutation rates for each trait
        debug_info : DebugInfo 
            Determines how much information should be printed to the console during the program's execution (default is no debug info)
        max_days : optional(int)
            If not None, this is the maximum number of days the simulation can run for before being automatically terminated (default is None)

        Returns
        -------
        days_survived : int 
            The number of days the species has survived until extinction 
        log : list(LogItem)
            A list of log item entries (important values for emulation training: see world.LogItem) made throughout the simulation's execution
        """
        days_survived, log = self.world.run(
            mutation_rates, debug_info, max_days)

        return days_survived, log

    def run_from_start_point(self, mutation_rates, day_start_point, population_start_point, mutation_start_point, debug_info=DebugInfo(), max_days=None):
        """
        Run the simulation from a starting point until the species goes extinct. 
        We are given the fixed mutation rates for the entire simulation. And the day, population and mean + std of each trait at the simulation start point.

        Parameters
        ----------
        mutation_rates : dict(string, int)
            Contains keys: size, speed, vision, aggression. 
            With corresponding values representing the mutation rates for each trait
        day_start_point : int 
            The day in which the simulation starts from (assert that 0 <= day_start_point and day_start_point < max_days if max_days is not None)
        population_start_point : int 
            The population at the simulation start point (assert that 0 <= population_start_point <= (self.grid_length_size ** 2))        
        mutation_start_point : dict(string, (int, int))
            Contains keys: size, speed, vision, aggression. 
            With corresponding values (mean, std) for the current mean and std of each trait
        debug_info : DebugInfo 
            Determines how much information should be printed to the console during the program's execution (default is no debug info)
        max_days : optional(int)
            If not None, this is the maximum number of days the simulation can run for before being automatically terminated (default is None)

        Returns
        -------
        days_survived : int 
            The number of days the species has survived until extinction 
        log : list(LogItem)
            A list of log item entries (important values for emulation training: see world.LogItem) made throughout the simulation's execution

        """

        assert (0 <= day_start_point)

        if max_days:
            assert (day_start_point < max_days)

        assert (0 <= population_start_point)
        assert (population_start_point <= (self.grid_length_size ** 2))

        days_survived, log = self.world.run_from_start_point(
            mutation_rates, day_start_point, population_start_point, mutation_start_point, debug_info, max_days)

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
