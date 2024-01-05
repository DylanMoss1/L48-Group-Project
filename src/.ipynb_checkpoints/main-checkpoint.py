from simulator import MainSimulator, TinySimulator
from world import DebugInfo
from pprint import pprint


if __name__ == "__main__":

    main_simulator = MainSimulator()

    mutation_rates = {
        "size": 1,
        "speed": 1,
        "vision": 1,
        "aggression": 1
    }

    days_survived, log = main_simulator.run(mutation_rates, debug_info=DebugInfo(
        period=10, should_display_day=True, should_display_grid=False, should_display_traits=False), max_days=100)

    # days_survived, log = main_simulator.run(mutation_rates, max_days=1000)

    print(days_survived)

    for log_item in log:
        print(log_item)
