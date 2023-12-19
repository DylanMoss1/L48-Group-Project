# import random
# from typing import Optional
# from tabulate import tabulate
# import math

# # --- Helper Functions ---

# species_id = 0


# def get_new_species_id() -> int:  # Returns 0, 1, 2, ... on consecutive function calls
#   global species_id
#   species_id += 1
#   return species_id - 1

# # --- Main Classes ---


# class World:
#   def __init__(self) -> None:
#     self.grid_size = 20
#     self.num_initial_species = 5
#     self.days = 0

#     self.species_grid = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]  # Generate N x N grid of None
#     self.populate_species_grid()

#     self.food_grid = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]  # Generate N x N grid of None
#     self.add_food_to_food_grid()

#   def compute_temperature(self):
#     # Compute temperature based off of days

#     # TODO: add climate change
#     return 8 - 18 * math.cos(2 * math.pi * self.days / 365)  # From Chapter 4.2 of https://link.springer.com/article/10.1007/s11538-008-9389-z

#   def populate_species_grid(self) -> None:
#     # Fill self.species_grid with self.num_initial_species species instances

#     species_location_set = set()  # Using sets so that locations are unique

#     while len(species_location_set) < self.num_initial_species:  # Keep adding locations until the set contains 4 unique locations
#       random_tuple = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
#       species_location_set.add(random_tuple)

#     for species_x, species_y in species_location_set:  # Add new species instances at every location in the set
#       self.species_grid[species_y][species_x] = Species()

#   def add_food_to_food_grid(self):
#     # Add food to self.food_grid based on current temperate

#     self.temperate = self.compute_temperature()



#   def pprint(self, display_grid=True, display_traits=True) -> None:
#     # Pretty print self.species_grid to console

#     # Print out species on the grid
#     if display_grid:
#       for row in self.species_grid:
#         for optional_species in row:
#           if optional_species:
#             species = optional_species
#             print(str(species.id).rjust(2), end="   ")
#           else:
#             print("-".rjust(2), end="   ")
#         print()

#       print("\n")

#     # Print out traits of living species in a table
#     if display_traits:

#       # Find all species in the grid
#       species_traits_list = []

#       for row in self.species_grid:
#         for optional_species in row:
#           if optional_species:
#             species = optional_species
#             species_traits_list.append([species.id, species.size, species.speed, species.vision, species.aggression])

#       species_traits_list.sort(key=lambda species_traits: species_traits[0])

#       table = [['Species ID', 'Size', 'Speed', 'Vision', 'Aggression']] + species_traits_list

#       print(tabulate(table, headers='firstrow', tablefmt='fancygrid'))


# class Species:
#   def __init__(self) -> None:
#     self.id = get_new_species_id()
#     self.size = 0
#     self.speed = 0
#     self.vision = 0
#     self.aggression = 0

# # --- Entry Point ---


# if __name__ == "__main__":
#   world = World()
#   world.pprint()
