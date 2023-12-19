import random
from species import Species
import math
from pprint import pprint
from tabulate import tabulate


class Food:
    """
    A class representing food on the simulation grid.

    Attributes
    ----------
    value : int 
        Represents how much food there is
    """

    def __init__(self) -> None:
        """
        Initialise a Food object.
        """
        self.value = 1


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
        species: Species
            The species object to be added to self.species_list
        """
        self.species_list.append(species)

    def get_species_id_list(self) -> list[Species]:
        """
        TODO: finish documentation
        """
        return list(map(lambda species: str(species.id), self.species_list))

    def get_food_value_list(self) -> list[Food]:
        """
        TODO: finish documentation
        """
        return list(map(lambda food: str(food.value), self.food_list))


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
        self.num_initial_species = 5
        self.days = 0

        self.grid = [
            [Location() for _ in range(self.grid_length_size)] for _ in range(self.grid_length_size)
        ]
        self.populate_grid()

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

        We compute seasonal variance as: 8 - 18cos(2π * days / 365) as seen in Chapter 4.2 of https://link.springer.com/article/10.1007/s11538-008-9389-z

        TODO: implement climate change         
        """

        return 8 - 18 * math.cos(2 * math.pi * self.days / 365)

    def add_food_to_grid(self) -> None:
        """
        Add food to grid depending on current temperature. 

        We model the probability of a food appearing in a location as a scaled Gaussian distribution:
            probability_of_food = scalar * exp(-0.5 * ((temperature - optimal_temperature) / sigma) ** 2)

        TODO: find a scientific backing behind how food availability depends on temperature  
        """

        optimal_temperature = 10  # TODO: find a better optimal_temperature value
        scalar = 0.1
        sigma = 1.0

        temperature = self.compute_temperature()

        return scalar * math.exp(-0.5 * ((temperature - optimal_temperature) / sigma) ** 2)

    def pprint(self, display_grid=True, display_traits=True) -> None:
        """
        Pretty print the World's current state. 

        Parameters
        ----------
        display_grid: bool
            If true, pretty print self.grid (default is True)
        display_traits: bool
            If true, pretty print all species traits in a table (default is True)
        """

        # Pretty print self.grid
        if display_grid:
            for row in self.grid:
                for location in row:
                    location_string = ""  # String to be printed out for this grid location
                    location_string += "/".join(location.get_species_id_list())
                    if location.species_list and location.food_list:
                        location_string += "||"
                    location_string += "/".join(location.get_food_value_list())
                    if location_string == "":
                        location_string = "-"
                    print(location_string.rjust(4), end=" " * 4)
                print()

        print("\n")

        # Pretty print all species traits in a table
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
