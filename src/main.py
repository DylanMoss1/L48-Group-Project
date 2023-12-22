from simulator import MainSimulator, TinySimulator
from world import DebugInfo


if __name__ == "__main__":

    tiny_simulator = TinySimulator()

    mutation_rates = {
        "size": 1,
        "speed": 1,
        "vision": 1,
        "aggression": 1
    }

    days_survived, log = tiny_simulator.run(mutation_rates, debug_info=DebugInfo(
        period=1, should_display_grid=True, should_display_traits=True))

    # main_simulator = MainSimulator()

    # days_survived, log = main_simulator.run(mutation_rates)
