import random
from species import Species
from food import Food
import math
from pprint import pprint
from tabulate import tabulate
from termcolor import colored


class Location:
    """
    A class representing a location on the simulation grid.

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


class World:
    """
    A class representing the simulation world.

    Attributes
    ----------
    grid_length_size : constant(int)
        Length of each size of the simulation grid.
    num_inital_species : constant(int)
        Number of inital species placed onto the grid
    days : int
        Days elapsed since the start of the simulation (starts at 0)
    grid : list(list(Location))
        Stores the current state of the world. A grid_size x grid_size matrix of Location instances
    """

    def __init__(self) -> None:
        """
        Initialise the World object.
        """
        self.grid_length_size = 20
        self.num_initial_species = 50
        self.days = 0

        self.grid = [
            [Location() for _ in range(self.grid_length_size)] for _ in range(self.grid_length_size)
        ]
        self.populate_grid()
        self.add_food_to_grid()  # TODO: remove

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

        return 8 - 18 * math.cos(2 * math.pi * self.days / 365)

    def add_food_to_grid(self) -> None:
        """
        Add food to grid depending on current temperature. 

        We model the probability of a food appearing in a location as a scaled Gaussian distribution:
            probability_of_food = scalar * exp(-0.5 * ((temperature - optimal_temperature) / sigma) ** 2)

        TODO: find a scientific backing behind how food availability depends on temperature  

        Returns 
        -------
        probability_of_food : float
            Probability of a food being generated in any location 
        """
        optimal_temperature = 10  # TODO: find a better optimal_temperature value
        scalar = 0.1
        sigma = 10

        temperature = self.compute_temperature()
        probability_of_food = scalar * \
            math.exp(-0.5 * ((temperature - optimal_temperature) / sigma) ** 2)

        for row in self.grid:
            for location in row:
                if random.random() < probability_of_food:
                    location.add_food()

        return probability_of_food

    def pprint(self, display_grid=True, display_traits=True) -> None:
        """
        Pretty print the World's current state. 

        Parameters
        ----------
        display_grid : bool
            If true, pretty print self.grid (default is True).
            Each location is represented as s1,...,sn||f1,...,fn for species ids s1,...,sn and food values f1,...,fn in each location.
            E.g. 1,3||2,2 represents species objects with ids 1 and 3, and food objects of values 2 and 2, occupying this location
        display_traits : bool
            If true, pretty print all traits of living species in a table (default is True)
        """

        # Pretty print self.grid
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

            print(tabulate(pprint_grid, tablefmt='rounded_grid', stralign='center'))
            print("\n")

        # Pretty print all traits of living species in a table
        if display_traits:
            species_traits_list = []

            for row in self.grid:
                for location in row:
                    for species in location.species_list:
                        species_traits_list.append(
                            [species.id, species.size, species.speed, species.vision, species.aggression])

            species_traits_list.sort(
                key=lambda species_traits: species_traits[0])

            table = [['Species ID', 'Size', 'Speed',
                      'Vision', 'Aggression']] + species_traits_list

            print(tabulate(table, headers='firstrow', tablefmt='fancygrid'))
