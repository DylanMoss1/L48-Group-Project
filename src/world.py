import random
from typing import Dict, Any, List
from species import Species
from food import Food
import math
from pprint import pprint
from tabulate import tabulate
from termcolor import colored
import constants
from dataclasses import dataclass
from numpy.random import normal


class Location:
    """
    A class representing a location on the simulation grid. This location can store multiple Species objects and Food objects.

    Attributes
    ----------
    species_list : list(Species)
        Stores a list of all species currently in this location
    food_list : list(Food)
        Stores a list of all food currently in this location
    """

    def __init__(self) -> None:
        """
        Initialise a Location object.
        """
        self.species_list = []
        self.food_list = []

    def add_species(self, species: Species) -> None:
        """
        Add a species object to self.species_list.

        Parameters
        ----------
        species : Species
            The species object to be added to self.species_list
        """
        self.species_list.append(species)

    def add_food(self) -> None:
        """
        Add a new food object to self.food_list.
        """
        self.food_list.append(Food())

    def get_species_id_list(self) -> list[Species]:
        """
        Returns a list containing string(species.id) for each species in self.species_list.
        """
        return [str(species.id) for species in self.species_list]

    def get_food_value_list(self) -> list[Food]:
        """
        Returns a list containing string(food.value) for each food in self.food_list.
        """
        return [str(food.value) for food in self.food_list]


@dataclass
class DebugInfo:
    """
    A class containing information about how much information should be printed to the console during the program's execution.
    The default is no debug info.

    Attributes
    ----------
    period : int
        The number of days to wait before printing debug info (default is 1)
    should_display_day : bool
        Should display the current day of the simulation (default is False)
    should_display_action : bool
        Should display the number of actions taken so far that day (default is False)
    should_display_grid : bool
        Should the grid state be printed in the debug info (default is False)
    should_display_traits : bool
        Should the traits of each species be printed in the debug info (default is False)
    should_display_population : bool
        Should display the total number of living species on this given day (default is False)
    """

    period: int = 1
    should_display_day: bool = False
    should_display_action: bool = False
    should_display_grid: bool = False
    should_display_traits: bool = False
    should_display_population: bool = False


@dataclass
class LogItem:
    """
    Contains a log entries for each day of the simulation, for analysis and training emulators.

    Attributes
    ----------
    day : int
        The current day of the simulation
    num_species_alive : int
        The number of species that still live at the *start* of this day
    temperature : float
        Current temperature on this day
    probability_of_food : float
        Probability of a food being generated in any location on this day
    traits_dict : Dict[str, List[float]]
        Contains traits of all living species at the *start* of this day in the form:
        {
          "size" : [size_species_1, ..., size_species_n],
          "speed" : [speed_species_1, ..., speed_species_n],
          "vision" : [vision_species_1, ..., vision_species_n],
          "aggression" : [aggression_species_1, ..., aggression_species_n],
          "energy": [energy_species_1, ..., energy_species_n],
        }
    """

    day: int
    num_species_alive: int
    temperature: float
    probability_of_food: float
    traits_dict: Dict[str, List[float]]


class World:
    """
    A class representing the simulation world.

    Parameters
    ----------
    grid_length_size : int
        Length of each size of the simulation grid.

    Attributes
    ----------
    directions : static(list(str))
        Directions that the species can move in on the grid: ['N', 'S', 'W', 'E']
    directions_to_location_change : static(dict(str, tuple(int, int)))
        Maps directions to a location change tuple. E.g. 'N' -> (-1, 0) for 1 step up the 2D array and 0 steps to the right

    grid_length_size : int
        Length of each size of the simulation grid.
    num_inital_species : constant(int)
        Number of inital species placed onto the grid
    num_actions_per_day : constant(int)
        Number of moving and feeding actions made per day
    days : int
        Days elapsed since the start of the simulation (starts at 0)
    grid : list(list(Location))
        Stores the current state of the world. A grid_size x grid_size matrix of Location instances
    """

    directions = ["N", "S", "W", "E"]
    directions_to_location_change = {
        "N": (-1, 0),
        "S": (1, 0),
        "E": (0, 1),
        "W": (0, -1),
    }
    opposite_direction = {
        "N": "S",
        "S": "N",
        "E": "W",
        "W": "E",
    }

    def __init__(self, grid_length_size) -> None:
        """
        Initialise the World object.
        """

        self.grid_length_size = grid_length_size

        self.num_initial_species = constants.NUM_INITIAL_SPECIES_FRACTION * (
            grid_length_size**2
        )

        self.day = 0
        self.num_actions_per_day = constants.NUM_ACTIONS_PER_DAY

        self.grid = [
            [Location() for _ in range(self.grid_length_size)]
            for _ in range(self.grid_length_size)
        ]

    def set_mutation_rates(self, mutation_rates) -> None:
        """
        Set the mutation rate values of the World instance.

        Attributes
        ----------
        size_mutation_rate : int
            Mutation rate for species size: size_{t+1} = N(size_t, size_mutation_rate)
        speed_mutation_rate : int
            Mutation rate for species speed: speed_{t+1} = N(speed_t, speed_mutation_rate)
        vision_mutation_rate : int
            Mutation rate for species vision: vision_{t+1} = N(vision_t, vision_mutation_rate)
        aggression_mutation_rate : int
            Mutation rate for species aggression: aggression_{t+1} = N(aggression_t, aggression_mutation_rate)
        """

        self.size_mutation_rate = mutation_rates["size"]
        self.speed_mutation_rate = mutation_rates["speed"]
        self.vision_mutation_rate = mutation_rates["vision"]
        self.aggression_mutation_rate = mutation_rates["aggression"]
        self.mutation_rates = {
            "size": mutation_rates["size"],
            "speed": mutation_rates["speed"],
            "vision": mutation_rates["vision"],
            "aggression": mutation_rates["aggression"],
        }

    def run_simulation_loop(self, debug_info=DebugInfo(), max_days=None):
        """
        Run the World simulation until the species goes extinct.

        Parameters
        ----------
        debug_info : DebugInfo
            Determines how much information should be printed to the console during the program's execution (default is no debug info)
        max_days : optional(int)
            If not None, this is the maximum number of days the simulation can run for before being automatically terminated (default is None)

        Returns
        -------
        days_survived : int
            The number of days the species has survived until extinction
        log : list(LogItem)
            A list of log item entries (important values for emulation training: see LogItem) made throughout the simulation's execution
        """

        log = []
        is_extinct = False

        while not is_extinct:
            self.day += 1

            if max_days:
                if self.day > max_days:
                    self.day -= 1
                    break

            is_extinct, log_item = self.compute_timestep(debug_info)
            log.append(log_item)
            self.debug(debug_info)

        days_survived = self.day

        return days_survived, log

    def run(
        self, mutation_rates, debug_info=DebugInfo(), max_days=None
    ) -> (int, List[LogItem]):
        """
        Run the World simulation with given mutation rates until the species goes extinct.

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
            A list of log item entries (important values for emulation training: see LogItem) made throughout the simulation's execution
        """

        self.populate_grid()
        self.day = 0

        self.set_mutation_rates(mutation_rates)
        return self.run_simulation_loop(debug_info, max_days)

    def run_from_start_point(
        self,
        mutation_rates,
        day_start_point,
        population_start_point,
        mutation_start_point,
        debug_info=DebugInfo(),
        max_days=None,
    ) -> (int, List[LogItem]):
        """
        Run the World simulation from a start point with given mutation rates until the species goes extinct.
        The current day, population and mean + std of each mutation trait are given for the start point.

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
            A list of log item entries (important values for emulation training: see LogItem) made throughout the simulation's execution

        Attributes
        ----------
        size_mutation_rate : int
            Mutation rate for species size: size_{t+1} = N(size_t, size_mutation_rate)
        speed_mutation_rate : int
            Mutation rate for species speed: speed_{t+1} = N(speed_t, speed_mutation_rate)
        vision_mutation_rate : int
            Mutation rate for species vision: vision_{t+1} = N(vision_t, vision_mutation_rate)
        aggression_mutation_rate : int
            Mutation rate for species aggression: aggression_{t+1} = N(aggression_t, aggression_mutation_rate)
        """

        self.populate_grid_from_start_point(
            population_start_point, mutation_start_point
        )
        self.day = day_start_point

        self.set_mutation_rates(mutation_rates)
        return self.run_simulation_loop(debug_info, max_days)

    def debug(self, debug_info) -> None:
        """
        If the current day reaches the debug_info period, pretty print the World's state according to the remaining debug_info parameters: should_display_grid and should_display_traits

        Parameters
        ----------
        debug_info : DebugInfo
            Determines how much information should be printed to the console during the program's execution (default is no debug info)
        """
        if self.day % debug_info.period == 0:
            if debug_info.should_display_day:
                print("Day number:", self.day)

            if debug_info.should_display_population:
                print("Population:", self.count_living_species())

            self.pprint(
                debug_info.should_display_grid,
                debug_info.should_display_traits,
            )

    def compute_timestep(self, debug_info) -> None:
        """
        Perform a timestep (that is process 1 day) of the World simulation.

        Parameters
        ----------
        debug_info : DebugInfo
            Determines how much information should be printed to the console during the program's execution (default is no debug info)

        Returns
        -------
        is_extinct : bool
            This is true if, and only if, all species have died
        log_item : LogItem
            The log item entry for this timestep of the simulation (important values for emulation training: see LogItem)
        """

        traits_dict = self.get_traits_of_living_species()
        temperature, probability_of_food = self.add_food_to_grid()
        self.species_hibernate()

        for action_number in range(self.num_actions_per_day):
            if debug_info.should_display_action:
                self.pprint(
                    debug_info.should_display_grid,
                    debug_info.should_display_traits,
                )

            self.species_move(action_number)
            self.species_consume_food()

        self.species_lose_energy()
        self.species_reproduce()
        self.species_age()
        num_species_alive = self.species_die()

        log_item = LogItem(
            self.day,
            num_species_alive,
            temperature,
            probability_of_food,
            traits_dict,
        )

        is_extinct = num_species_alive == 0

        return is_extinct, log_item

    def populate_grid(self) -> None:
        """
        Fill self.grid with self.num_inital_species number of initial species.
        """

        species_location_set = set()  # Using sets so that locations are unique

        # Keep adding locations until the set contains 4 unique locations
        while len(species_location_set) < self.num_initial_species:
            random_tuple = (
                random.randint(0, self.grid_length_size - 1),
                random.randint(0, self.grid_length_size - 1),
            )
            species_location_set.add(random_tuple)

        for species_x, species_y in species_location_set:
            # Add new species instances at every location in the set
            self.grid[species_y][species_x].add_species(Species())

    def populate_grid_from_start_point(
        self, population_start_point, mutation_start_point
    ) -> None:
        """
        Fill self.grid with population_start_point number of initial species.
        The mutations of these species are described by the mean and std in mutation_start_point

        Parameters
        ----------
        population_start_point : int
            The population at the simulation start point (assert that 0 <= population_start_point <= (self.grid_length_size ** 2))
        mutation_start_point : dict(string, (int, int))
            Contains keys: size, speed, vision, aggression.
            With corresponding values (mean, std) for the current mean and std of each trait
        """

        species_location_set = set()  # Using sets so that locations are unique

        # Keep adding locations until the set contains 4 unique locations
        while len(species_location_set) < population_start_point:
            random_tuple = (
                random.randint(0, self.grid_length_size - 1),
                random.randint(0, self.grid_length_size - 1),
            )
            species_location_set.add(random_tuple)

        for species_x, species_y in species_location_set:
            # Add new species instances at every location in the set
            size = normal(
                loc=mutation_start_point["size"][0],
                scale=mutation_start_point["size"][1],
            )
            speed = normal(
                loc=mutation_start_point["speed"][0],
                scale=mutation_start_point["speed"][1],
            )
            vision = normal(
                loc=mutation_start_point["vision"][0],
                scale=mutation_start_point["vision"][1],
            )
            aggression = normal(
                loc=mutation_start_point["aggression"][0],
                scale=mutation_start_point["aggression"][1],
            )

            self.grid[species_y][species_x].add_species(
                Species(
                    size=size,
                    speed=speed,
                    vision=vision,
                    aggression=aggression,
                )
            )

    def get_traits_of_living_species(self) -> Dict[str, List[float]]:
        """
        Returns each trait of all living species in list form, accessible through a dictionary.

        Returns
        -------
        traits_dict : Dict[str, List[float]]
            Contains traits of all living species in the form:
            {
              "size" : [size_species_1, ..., size_species_n],
              "speed" : [speed_species_1, ..., speed_species_n],
              "vision" : [vision_species_1, ..., vision_species_n],
              "aggression" : [aggression_species_1, ..., aggression_species_n],
              "energy" : [energy_species_1, ..., energy_species_n],
            }
        """

        traits_dict = {
            "size": [],
            "speed": [],
            "vision": [],
            "aggression": [],
            "energy": [],
        }

        for row in self.grid:
            for location in row:
                for species in location.species_list:
                    traits_dict["size"].append(species.size)
                    traits_dict["speed"].append(species.speed)
                    traits_dict["vision"].append(species.vision)
                    traits_dict["aggression"].append(species.aggression)
                    traits_dict["energy"].append(species.energy)

        return traits_dict

    def compute_temperature(self) -> int:
        """
        Compute the temperature depending on the day.

        This takes into account both (periodic) seasonal variance and (linear) climate change.

        We compute seasonal variance as: 8 - 18cos(2π * days / 365) as seen in Chapter 4.2 of https://link.springer.com/article/10.1007/s11538-008-9389-z.

        Returns
        -------
        temperature : float
            Temperature on the current day
        """

        return (
            10 + 18 * math.sin(2 * math.pi * self.day / 100) + (self.day / 100)
        )

    def add_food_to_grid(self) -> None:
        """
        Add food to grid depending on current temperature.

        We model the probability of a food appearing in a location as a scaled Gaussian distribution:
            probability_of_food = scalar * exp(-0.5 * ((temperature - optimal_temperature) / sigma) ** 2)

        Returns
        -------
        temperature : float
            Current temperature on this day
        probability_of_food : float
            Probability of a food being generated in any location on this day
        """

        optimal_temperature = constants.OPTIMAL_TEMPERATURE
        scalar = constants.FOOD_PROBABILITY_SCALAR
        sigma = constants.FOOD_PROBABILITY_STD

        temperature = self.compute_temperature()

        probability_of_food = scalar * math.exp(
            -0.5 * (abs(temperature - optimal_temperature) / sigma) ** 2
        )

        for row in self.grid:
            for location in row:
                if random.random() < probability_of_food:
                    location.add_food()

        return temperature, probability_of_food

    def add_species_to_grid(self, species, row_index, col_index) -> None:
        """
        Add a given species to the grid at a given row and column index.

        Parameters
        ----------
        species : Species
            Species to be added to grid
        row_index : int
            Index of grid row where the species should be added
        col_index : int
            Index of grid col where the species should be added
        """
        self.grid[row_index][col_index].add_species(species)

    def is_valid_location(self, location) -> bool:
        """
        Returns true if a given location (in form (row_index, col_index)) is within the boundaries of the grid.

        Parameters
        ----------
        location : (int, int)
            Location represented in form (row_index, col_index)

        Returns
        -------
        is_valid_location : bool
            Is true if a given location (in form (row_index, col_index)) is within the boundaries of the grid
        """

        row_index, col_index = location

        if col_index < 0 or col_index >= self.grid_length_size:
            return False

        if row_index < 0 or row_index >= self.grid_length_size:
            return False

        return True

    def is_food_at_location(self, location) -> bool:
        """
        Returns true if the given location (in form (row_index, col_index)) contains food.

        Parameters
        ----------
        location : (int, int)
            Location represented in form (row_index, col_index)

        Returns
        -------
        is_food : bool
            Is true if a given location (in form (row_index, col_index)) on the grid contains food
        """

        row_index, col_index = location
        food_list_at_location = self.grid[row_index][col_index].food_list
        is_food = len(food_list_at_location) > 0

        return is_food

    def food_directions_found_in_vision(
        self,
        possible_directions,
        current_species_row_index,
        current_species_col_index,
        current_vision,
    ) -> List[str]:
        """
        Returns a list of all directions (i.e. "N", "S", "E", or "W") of food found with a set vision.
        Note that we only look at positions EXACTLY as far as the vision, not any less than the vision.

        Parameters
        ----------
        possible_directions : list(str)
            A subset of World.directions (['N', 'S', 'W', 'E']), containing all directions the species can possibly move in
        current_species_row_index : int
            Current row index of the species on the grid
        current_species_col_index : int
            Current col index of the species on the grid
        current_vision : int
            The current vision range we are examining (we only look at locations EXACTLY this far)

        Returns
        -------
        food_locations : list(str)
            List of all directions containing food found with the set vision
        """

        food_directions = []

        for direction in possible_directions:
            row_change, col_change = World.directions_to_location_change[
                direction
            ]

            # The indices of the location the species is currently looking at (to find food)

            currently_observed_row_index = (
                current_species_row_index + row_change * current_vision
            )
            currently_observed_col_index = (
                current_species_col_index + col_change * current_vision
            )

            currently_observed_location = (
                currently_observed_row_index,
                currently_observed_col_index,
            )

            if self.is_valid_location(
                currently_observed_location
            ) and self.is_food_at_location(currently_observed_location):
                food_directions.append(direction)

        return food_directions

    def is_food_at_location(self, currently_observed_location) -> bool:
        row_index, col_index = currently_observed_location
        return len(self.grid[row_index][col_index].food_list) > 0

    def decide_direction(
        self, species, species_location, possible_directions
    ) -> str:
        """
        Given the possible directions that a species can take, determine the best direction they should move in.
        - This should be towards the direction with the closest food in the species' vision.
        - Ties should be broken randomly.
        - Vision only works along rows and columns, not diagonally.

        Vision of 3.4:
        - Look 3 spaces in all possible directions
        - Move towards closest food found
        - If no food found, look 4 spaces in all possible directions
        - If food found:
          - 40% chance (3.4 - 3 = 0.4): Move towards food
          - 60% chance: Move in a random possible direction
        - If no food found
          - Move in a random possible direction

        Parameters
        ----------
        species : Species
            Current species that is looking to move.
        species_location : tuple(int, int)
            Current location of the species that is looking to move
        possible_directions : list(str)
            A subset of World.directions (['N', 'S', 'W', 'E']), containing all directions the species can possibly move in

        Returns
        -------
        best_direction : str
            The best direction the species can move in (based off the food found with the species' vision)
        """
        parameter = constants.MAXIMUM_VISION
        sight = species.vision
        current_species_row_index, current_species_col_index = species_location
        vision = self.grid_length_size * sight * parameter
        # Look {vision // 1} spaces in all possible directions
        for current_vision in range(1, math.floor(vision) + 1):
            food_directions = self.food_directions_found_in_vision(
                possible_directions,
                current_species_row_index,
                current_species_col_index,
                current_vision,
            )

            # Return the closest food found
            # Ties broken randomly
            if len(food_directions) > 0:
                return random.choice(food_directions)

        # If no food found so far...
        # Look {vision // 1} + 1 spaces in all possible directions
        food_directions = self.food_directions_found_in_vision(
            possible_directions,
            current_species_row_index,
            current_species_col_index,
            current_vision=math.floor(vision) + 1,
        )

        # If food found, move towards food with (vision - vision // 1) probability -- that is with probability U[0, 1] < (vision % 1)
        if len(food_directions) > 0 and random.random() < (vision % 1):
            return random.choice(food_directions)
        else:
            return random.choice(possible_directions)

    def species_move(self, action_number) -> None:
        """
        All living species make a move on the grid.

        For all species
        - 1) If it is able to move (according to its speed) and it has not previously moved this action_number...
        - 2) Determine what directions it can move in (not off the grid)
        - 3) Figure out the best direction the species can move (according to its vision)
        - 4) Set this as the species' last moved direction, and make a move in this direction

        Parameters
        ----------
        action_number : int
            The number of actions that have already happened this day
        """

        """
        Explanation: The code has been changed to use a slowness factor instead. 
        The fastest creature statisfies the modulo able_to_move condition at all times with slowness factor 1. 
        A creature which is half as fast will satisfy the able_to_move condition half the action steps with slowness factor
        2 and so on
        """

        maximum_speed = constants.MAXIMUM_SPEED

        moved_species = []

        for row_index, row in enumerate(self.grid):
            for col_index, location in enumerate(row):
                for species in location.species_list:
                    slowness_factor = (
                        (int)(maximum_speed / species.speed) + 1
                        if species.speed > 0
                        else -1
                    )

                    able_to_move = (
                        action_number % slowness_factor == 0
                        and (not species.hibernate)
                        and (slowness_factor != -1)
                    )

                    has_previously_moved = species.id in moved_species

                    # 1) If it is able to move (according to its speed) and it has not previously moved this action_number...
                    if able_to_move and not has_previously_moved:
                        # 2) Determine what directions it can move in (not off the grid
                        remaining_directions = World.directions.copy()

                        invalid_directions = []

                        # Remove all directions where the species goes off the grid
                        for direction in remaining_directions:
                            (
                                row_change,
                                col_change,
                            ) = World.directions_to_location_change[direction]

                            if not self.is_valid_location(
                                (
                                    row_index + row_change,
                                    col_index + col_change,
                                )
                            ):
                                invalid_directions.append(direction)

                        for direction in invalid_directions:
                            remaining_directions.remove(direction)

                        # 3) Figure out the best direction the species can move (according to its vision)
                        new_direction = self.decide_direction(
                            species,
                            (row_index, col_index),
                            remaining_directions,
                        )

                        # 4) Set this as the species' last moved direction, and make a move in this direction
                        species.last_moved_direction = new_direction

                        (
                            row_change,
                            col_change,
                        ) = World.directions_to_location_change[new_direction]

                        self.add_species_to_grid(
                            species,
                            row_index + row_change,
                            col_index + col_change,
                        )

                        location.species_list.remove(species)
                        moved_species.append(species.id)

    def species_consume_food(self) -> None:
        """
        If species are in a location with food, they consume all the food to gain energy.

        If more than one species are in the same location with food, they share or fight over the food according to their aggression metrics.
        @Atreyi to explain in more detail

        The dove to hawk threshold is 0.5
        """

        food_value = constants.FOOD_VALUE
        damage_value = constants.DAMAGE_VALUE

        for row in self.grid:
            for location in row:
                if (
                    len(location.species_list) > 0
                    and len(location.food_list) > 0
                ):
                    if len(location.species_list) == 1:
                        for species in location.species_list:
                            species.energy += (
                                len(location.food_list) * food_value
                            )
                    else:
                        aggression = [
                            species.aggression
                            for species in location.species_list
                        ]
                        if all(aggr <= 0.5 for aggr in aggression):
                            for species in location.species_list:
                                species.energy += (
                                    len(location.food_list)
                                    * food_value
                                    / len(location.species_list)
                                )
                        else:
                            winner_hawk_indices = [
                                i
                                for i, j in enumerate(aggression)
                                if j == max(aggression)
                            ]
                            if len(winner_hawk_indices) == 1:
                                max_damage = max(
                                    [
                                        i
                                        for i in aggression
                                        if i < max(aggression)
                                    ]
                                )
                                if max_damage <= 0.5:
                                    max_damage = 0
                            else:
                                max_damage = max(aggression)
                            winner_hawk = location.species_list[
                                random.sample(winner_hawk_indices, 1)[0]
                            ].id
                            for species in location.species_list:
                                if species.aggression > 0.5:
                                    if species.id == winner_hawk:
                                        species.energy += (
                                            len(location.food_list)
                                            * food_value
                                        )
                                        species.energy -= (
                                            max_damage * damage_value
                                        )
                                    else:
                                        species.energy -= (
                                            species.aggression * damage_value
                                        )
                    location.food_list = []

    def species_hibernate(self) -> None:
        """
        At the beginning of each day each creature may hibernate with a probability proportional to its size
        """
        for row in self.grid:
            for location in row:
                for species in location.species_list:
                    hibernation_risk = species.size
                    is_hibernating = random.random() < hibernation_risk
                    species.hibernate = is_hibernating

    def species_lose_energy(self) -> None:
        """
        Each species loses a set amount of energy at the end of every day.
        This energy loss is proportional to the square of the speed of the species.
        Species lose an additional amount of energy proportional to their vision trait
        Maximum energy is capped at a value that increases with the size of the species
        """

        energy_loss_base = constants.ENERGY_LOSS
        food_value = constants.FOOD_VALUE
        reproduction_threshold = constants.REPRODUCTION_THRESHOLD

        for row in self.grid:
            for location in row:
                for species in location.species_list:
                    energy_loss = ((1 + species.speed) ** 2) * energy_loss_base
                    energy_loss += ((species.vision) * energy_loss_base) / 2
                    species.energy -= energy_loss
                    maximum_stored_energy = (
                        reproduction_threshold + species.size * food_value * 10
                    )
                    species.energy = min(species.energy, maximum_stored_energy)

    def species_age(self) -> None:
        """
        all creatures get older
        """

        for row in self.grid:
            for location in row:
                for species in location.species_list:
                    species.age += 1

    def species_reproduce(self) -> None:
        """
        If a species has more than N energy, they reproduce asexually. The new species has mutated traits, distributed as Normal(μ=parent_trait, σ=trait_mutation_rate)
        """

        reproduction_threshold = constants.REPRODUCTION_THRESHOLD

        for row in self.grid:
            for location in row:
                for species in location.species_list:
                    if species.energy >= reproduction_threshold:
                        species.energy = (
                            species.energy - reproduction_threshold / 2
                        )

                        # Get parent's traits
                        parent_traits = species.get_traits()

                        # Add random mutation to generate child's trait values
                        child_traits = Species.get_child_traits(
                            parent_traits, self.mutation_rates
                        )

                        # Generate a child from these trait values
                        if child_traits is not None:
                            location.add_species(
                                Species(
                                    size=child_traits["size"],
                                    speed=child_traits["speed"],
                                    vision=child_traits["vision"],
                                    aggression=child_traits["aggression"],
                                    energy=reproduction_threshold / 2,
                                )
                            )

    def species_die(self) -> int:
        """
        If any species has less than or equal to 0 energy, they die.

        Returns
        -------
        num_species_alive : int
            The number of species that are still alive in the simulation World
        """

        num_alive_species = 0  # Calculated like this for the logs
        maximum_age = constants.MAXIMUM_AGE

        for row in self.grid:
            for location in row:
                dead_species_list = []

                for species in location.species_list:
                    if species.energy <= 0 or species.age >= maximum_age:
                        species.death = True
                        dead_species_list.append(species)
                    else:
                        num_alive_species += 1

                for species in dead_species_list:
                    location.species_list.remove(species)

        return num_alive_species

    def count_living_species(self) -> int:
        """
        Count the total number of species that are still alive in the simulation World.

        Returns
        -------
        num_species_alive : int
            The number of species that are still alive in the simulation World
        """

        num_alive_species = 0

        for row in self.grid:
            for location in row:
                num_alive_species += len(location.species_list)

        return num_alive_species

    def pprint(self, display_grid=True, display_traits=True) -> None:
        """
        Pretty print the World's current state.

        Parameters
        ----------
        display_grid : bool
            If true, pretty print self.grid (default is True).
        display_traits : bool
            If true, pretty print all traits of living species in a table (default is True)
        """

        # Pretty print self.grid
        #   Each location is represented as s1,...,sn||f1,...,fn for species ids s1,...,sn and food values f1,...,fn in each location.
        #   E.g. 1,3||2,2 represents species objects with ids 1 and 3, and food objects of values 2 and 2, occupying this location
        if display_grid:
            pprint_grid = [
                [None for _ in range(self.grid_length_size)]
                for _ in range(self.grid_length_size)
            ]

            for row_index, row in enumerate(self.grid):
                for col_index, location in enumerate(row):
                    pprint_location = ""
                    species_id_list = location.get_species_id_list()
                    food_value_list = location.get_food_value_list()

                    if species_id_list or food_value_list:
                        if species_id_list:
                            pprint_location += colored(
                                ",".join(species_id_list), "light_blue"
                            )
                        pprint_location += "||"
                        if food_value_list:
                            pprint_location += colored(
                                ",".join(food_value_list), "light_red"
                            )
                    else:
                        pprint_location = "⠀" * 3

                    pprint_grid[row_index][col_index] = pprint_location

            print(
                tabulate(
                    pprint_grid, tablefmt="rounded_grid", stralign="center"
                )
            )
            print("\n")

        # Pretty print all traits of living species in a table
        species_traits_list = []

        for row in self.grid:
            for location in row:
                for species in location.species_list:
                    species_traits_list.append(
                        [
                            species.id,
                            species.size,
                            species.speed,
                            species.vision,
                            species.aggression,
                            species.energy,
                        ]
                    )
        if display_traits:
            species_traits_list.sort(
                key=lambda species_traits: species_traits[0]
            )

            table = [
                [
                    "Species ID",
                    "Size",
                    "Speed",
                    "Vision",
                    "Aggression",
                    "Energy",
                ]
            ] + species_traits_list

            print(tabulate(table, headers="firstrow", tablefmt="fancygrid"))
            print("\n")
