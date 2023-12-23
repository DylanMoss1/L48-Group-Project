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
    should_display_grid : bool 
        Should the grid state be printed in the debug info (default is False) 
    should_display_traits : bool
        Should the traits of each species be printed in the debug info (default is False) 
    """
    period: int = 1
    should_display_day: bool = False
    should_display_grid: bool = False
    should_display_traits: bool = False


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

    def __init__(self, grid_length_size) -> None:
        """
        Initialise the World object.
        """

        self.grid_length_size = grid_length_size

        self.num_initial_species = constants.NUM_INITIAL_SPECIES_FRACTION * \
            (grid_length_size ** 2)

        self.day = 0
        self.num_actions_per_day = constants.NUM_ACTIONS_PER_DAY

        self.grid = [
            [Location() for _ in range(self.grid_length_size)] for _ in range(self.grid_length_size)
        ]
        self.populate_grid()

    def run(self, mutation_rates, debug_info=DebugInfo(), max_days=None) -> (int, List[LogItem]):
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
            "aggression": mutation_rates["aggression"]
        }

        self.day = 0
        log = []
        is_extinct = False

        while not is_extinct:
            self.day += 1

            if max_days:
                if self.day > max_days:
                    self.day -= 1
                    break

            is_extinct, log_item = self.compute_timestep()
            log.append(log_item)
            self.debug(debug_info)

        days_survived = self.day

        return days_survived, log

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

            self.pprint(debug_info.should_display_grid,
                        debug_info.should_display_traits)

    def compute_timestep(self) -> None:
        """
        Perform a timestep (that is process 1 day) of the World simulation.

        Returns 
        -------
        is_extinct : bool 
            This is true if, and only if, all species have died
        log_item : LogItem
            The log item entry for this timestep of the simulation (important values for emulation training: see LogItem)
        """

        self.day += 1

        traits_dict = self.get_traits_of_living_species()
        temperature, probability_of_food = self.add_food_to_grid()

        for action_number in range(self.num_actions_per_day):
            self.species_move(action_number)
            self.species_consume_food()

        self.species_lose_energy()
        num_species_alive = self.species_die()

        log_item = LogItem(self.day, num_species_alive,
                           temperature, probability_of_food, traits_dict)

        is_extinct = num_species_alive == 0

        return is_extinct, log_item

    def populate_grid(self) -> None:
        """
        Fill self.grid with self.num_inital_species number of initial species.
        """

        species_location_set = set()  # Using sets so that locations are unique

        # Keep adding locations until the set contains 4 unique locations
        while len(species_location_set) < self.num_initial_species:
            random_tuple = (random.randint(0, self.grid_length_size - 1),
                            random.randint(0, self.grid_length_size - 1))
            species_location_set.add(random_tuple)

        for species_x, species_y in species_location_set:
            # Add new species instances at every location in the set
            self.grid[species_y][species_x].add_species(Species())

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
            }
        """

        traits_dict = {"size": [], "speed": [], "vision": [], "aggression": []}

        for row in self.grid:
            for location in row:
                for species in location.species_list:
                    traits_dict["size"].append(species.size)
                    traits_dict["speed"].append(species.speed)
                    traits_dict["vision"].append(species.vision)
                    traits_dict["aggression"].append(species.aggression)

        return traits_dict

    def compute_temperature(self) -> int:
        """
        Compute the temperature depending on the day.

        This takes into account both (periodic) seasonal variance and (linear) climate change.

        We compute seasonal variance as: 8 - 18cos(2π * days / 365) as seen in Chapter 4.2 of https://link.springer.com/article/10.1007/s11538-008-9389-z.

        TODO: implement climate change

        Returns
        -------
        temperature : float
            Temperature on the current day
        """

        return 8 - 18 * math.cos(2 * math.pi * self.day / 365)

    def add_food_to_grid(self) -> None:
        """
        Add food to grid depending on current temperature.

        We model the probability of a food appearing in a location as a scaled Gaussian distribution:
            probability_of_food = scalar * exp(-0.5 * ((temperature - optimal_temperature) / sigma) ** 2)

        TODO: find a scientific backing behind how food availability depends on temperature

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

        probability_of_food = scalar * \
            math.exp(-0.5 * ((temperature - optimal_temperature) / sigma) ** 2)

        for row in self.grid:
            for location in row:
                if random.random() < probability_of_food:
                    location.add_food()

        return temperature, probability_of_food

    def add_species_to_grid(self, species, row_index, col_index):
        self.grid[row_index][col_index].add_species(species)

    def species_move(self, action_number) -> None:
        """
        All living species make a move on the grid.

        Note that this cannot be in the direction they moved last.

        Parameters
        ----------
        action_number : int 
            The number of actions that have already happened this day  
        """

        '''
        TODO: Add vision
        TODO: Add hibernate
        '''
        speed_modifier = constants.SPEED_MODIFIER
        directions = ['N', 'S', 'W', 'E']
        moved_species = []

        for row_index, row in enumerate(self.grid):
            for col_index, location in enumerate(row):
                for species in location.species_list:

                    able_to_move = action_number % (
                        species.speed) * speed_modifier == 0
                    has_previously_moved = species.id in moved_species

                    if able_to_move and not has_previously_moved:

                        directions_spec = random.sample(
                            directions, len(directions))

                        if species.last_moved_direction == 'N' or row_index == self.grid_length_size - 1:
                            directions_spec.remove('S')
                        if species.last_moved_direction == 'S' or row_index == 0:
                            directions_spec.remove('N')
                        if species.last_moved_direction == 'E' or col_index == self.grid_length_size - 1:
                            directions_spec.remove('E')
                        if species.last_moved_direction == 'W' or col_index == 0:
                            directions_spec.remove('W')

                        new_direction = directions_spec[0]
                        species.last_moved_direction = new_direction

                        match new_direction:
                            case 'N':
                                self.add_species_to_grid(
                                    species, row_index - 1, col_index)
                            case 'S':
                                self.add_species_to_grid(
                                    species, row_index + 1, col_index)
                            case 'W':
                                self.add_species_to_grid(
                                    species, row_index, col_index - 1)
                            case 'E':
                                self.add_species_to_grid(
                                    species, row_index, col_index + 1)

                        location.species_list.remove(species)
                        moved_species.append(species.id)

    def species_consume_food(self) -> None:
        """
        If species are in a location with food, they consume all the food to gain energy.

        If more than one species are in the same location with food, they share or fight over the food according to their aggression metrics.
        @Atreyi to explain in more detail  

        TODO: Add speed related energy waste
        """

        for row in self.grid:
            for location in row:
                if len(location.species_list) > 0 and len(
                        location.food_list) > 0:
                    aggression = [
                        species.aggression for species in location.species_list]
                    if all(aggr <= 1 for aggr in aggression):
                        for species in location.species_list:
                            species.energy += len(location.food_list) / \
                                len(location.species_list)
                    else:
                        winner_hawk_indices = [
                            i for i, j in enumerate(aggression)if j == max(aggression)]
                        if len(winner_hawk_indices) == 1:
                            max_damage = max(
                                [i for i in aggression if i < max(aggression)])
                            if max_damage <= 1:
                                max_damage = 0
                        else:
                            max_damage = max(aggression)
                        winner_hawk = location.species_list[random.sample(
                            winner_hawk_indices, 1)].id
                        for species in location.species_list:
                            if species.aggression > 1:
                                if species.id == winner_hawk:
                                    species.energy += len(location.food_list)
                                    species.energy -= max_damage / 2
                                else:
                                    species.energy -= species.aggression / 2
                    location.food_list = []

    def species_lose_energy(self) -> None:
        """
        Each species loses a set amount of energy at the end of every day. 
        """

        energy_loss = constants.ENERGY_LOSS

        for row in self.grid:
            for location in row:
                for species in location.species_list:
                    species.energy -= energy_loss

    def species_reproduce(self) -> None:
        """
        If a species has more than N energy, they reproduce asexually. The new species has mutated traits, distributed as Normal(μ=parent_trait, σ=trait_mutation_rate)
        """

        reproduction_threshold = constants.REPRODUCTION_THRESHOLD

        for row in self.grid:
            for location in row:
                for species in location.species_list:
                    if species.energy >= reproduction_threshold:

                        # Get parent's traits
                        parent_traits = species.get_traits()

                        # Add random mutation to generate child's trait values
                        child_traits = Species.get_child_traits(
                            parent_traits, self.mutation_rates)

                        # Generate a child from these trait values
                        location.add_species(Species(
                            size=child_traits["size"], speed=child_traits["speed"], vision=child_traits["vision"], aggression=child_traits["aggression"]))

    def species_die(self) -> bool:
        """
        If any species has less than or equal to 0 energy, they die.

        Returns
        -------
        num_species_alive : int
            The number of species that are still alive in the simulation World
        """

        num_alive_species = 0  # Calculated like this for the logs

        for row in self.grid:
            for location in row:
                for species in location.species_list:
                    if species.energy <= 0:
                        species.death = True
                        location.species_list.remove(species)
                    else:
                        num_alive_species += 1

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

            pprint_grid = [[None for _ in range(self.grid_length_size)] for _ in range(
                self.grid_length_size)]

            for row_index, row in enumerate(self.grid):
                for col_index, location in enumerate(row):
                    pprint_location = ""
                    species_id_list = location.get_species_id_list()
                    food_value_list = location.get_food_value_list()

                    if species_id_list or food_value_list:
                        if species_id_list:
                            pprint_location += colored(
                                ",".join(species_id_list), "light_blue")
                        pprint_location += "||"
                        if food_value_list:
                            pprint_location += colored(
                                ",".join(food_value_list), "light_red")
                    else:
                        pprint_location = "⠀" * 3

                    pprint_grid[row_index][col_index] = pprint_location

            print(
                tabulate(
                    pprint_grid,
                    tablefmt='rounded_grid',
                    stralign='center'))
            print("\n")

        # Pretty print all traits of living species in a table
        if display_traits:
            species_traits_list = []

            for row in self.grid:
                for location in row:
                    for species in location.species_list:
                        species_traits_list.append(
                            [species.id, species.size, species.speed, species.vision, species.aggression, species.energy])

            species_traits_list.sort(
                key=lambda species_traits: species_traits[0])

            table = [['Species ID', 'Size', 'Speed',
                      'Vision', 'Aggression', 'Energy']] + species_traits_list

            print(tabulate(table, headers='firstrow', tablefmt='fancygrid'))
            print("\n")
