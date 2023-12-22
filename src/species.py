import constants

species_id = 0


def get_new_species_id() -> int:
    """Returns a unique id (returns 0, 1, 2, ... on consecutive function calls)."""
    global species_id
    species_id += 1
    return species_id - 1


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

    def __init__(self) -> None:
        """
        Initialise a Species object.
        """

        self.id = get_new_species_id()
        self.size = constants.INITIAL_SIZE
        self.speed = constants.INITIAL_SPEED
        self.vision = constants.INITIAL_VISION
        self.aggression = constants.INITIAL_AGGRESSION
        self.energy = constants.INITIAL_ENERGY
        self.death = False
        self.last_moved_direction = None
