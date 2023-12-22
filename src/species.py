from typing import Dict
import constants
import random

# Generate unique species IDs

species_id = 0


def get_new_species_id() -> int:
    """Returns a unique id (returns 0, 1, 2, ... on consecutive function calls)."""
    global species_id
    species_id += 1
    return species_id - 1

# Collect relevant constants


initial_size = constants.INITIAL_SIZE
initial_speed = constants.INITIAL_SPEED
initial_vision = constants.INITIAL_VISION
initial_aggression = constants.INITIAL_AGGRESSION


class Species:
    """
    A class representing a species.

    Attributes
    ----------
    id : int
        Unique identifier for the species.
    size : float
        The size of the species.
    speed : float
        The speed of the species.
    vision : float
        The vision of the species.
    aggression : float
        The aggression of the species.
    energy : float 
        Represents how much energy the species has left, if this reaches 0 the species dies. 
    last_moved_direction : optional(int)  @Atreyi to update 
        Stores the direction the species last moved in.
        The value 0 represents North, 1 represents East, 2 represents South, 3 represents West.
    """

    def __init__(self, size=initial_size, speed=initial_speed, vision=initial_vision, aggression=initial_aggression) -> None:
        """
        Initialise a Species object.
        """

        self.id = get_new_species_id()
        self.size = size
        self.speed = speed
        self.vision = vision
        self.aggression = aggression
        self.energy = constants.INITIAL_ENERGY
        self.death = False
        self.last_moved_direction = None

    def get_traits(self) -> Dict[str, float]:
        """
        Get the species' traits in dictionary form.

        Returns
        -------
        traits : dict(str, float) 
            Traits in dictionary form (with keys "size", "speed", "vision", "aggression") 
        """
        return {
            "size": self.size,
            "speed": self.speed,
            "vision": self.vision,
            "aggression": self.aggression,
        }

    @staticmethod
    def get_child_traits(original_traits, mutation_rates) -> Dict[str, float]:
        """
        Add mutation to original_traits with mutation_rates to obtain the child trait values. 

        Parameters
        ----------
        original_traits : dict(str, float) 
            Parent traits in dictionary form (with keys "size", "speed", "vision", "aggression") 
        mutation_rates : dict(str, float)
            Global mutation rate values in dictionary form (with keys "size", "speed", "vision", "aggression") 

        Returns 
        -------
        new_traits : dict(str, float) 
            Child traits (parent traits + mutations) in dictionary form (with keys "size", "speed", "vision", "aggression")
        """

        new_traits = {"size": None, "speed": None,
                      "vision": None, "aggression": None}

        for key, value in original_traits.items():
            new_traits[key] = random.normal(
                loc=value, scale=mutation_rates[key])

        return new_traits
