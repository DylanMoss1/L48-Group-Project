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
    size : int
        The size of the species.
    speed : int
        The speed of the species.
    vision : int
        The vision of the species.
    aggression : int
        The aggression of the species.
    energy : int 
        Represents how much energy the species has left, if this reaches 0 the species dies. 
    has_moved : bool 
        Records whether the species has moved this timestep. 
    last_moved_direction : optional(int)
        Stores the direction the species last moved in.
        The value 0 represents North, 1 represents East, 2 represents South, 3 represents West.
    """

    def __init__(self) -> None:
        """
        Initialise a Species object.
        """

        self.id = get_new_species_id()
        self.size = 0
        self.speed = 0
        self.vision = 0
        self.aggression = 0
        self.energy = 5 
        self.has_moved = False
        self.last_moved_directon = None
