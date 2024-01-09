from simulator import MainSimulator, TinySimulator
from world import DebugInfo
from pprint import pprint


if __name__ == "__main__":

    main_simulator = MainSimulator()
    tiny_simulator = TinySimulator()

    mutation_rates = {
        "size": 0.2,
        "speed": 0.2,
        "vision": 0.2,
        "aggression": 0.2,
    }

    # mutation_start_point = {
    #     "size": (1, 0.5),
    #     "speed":  (1, 0.5),
    #     "vision":  (1, 0.5),
    #     "aggression": (1, 0.5),
    # }

    # days_survived, log = main_simulator.run(mutation_rates, debug_info=DebugInfo(
    #     period=10, should_display_day=True, should_display_grid=False, should_display_traits=False), max_days=100)

    # days_survived, log = tiny_simulator.run(mutation_rates, debug_info=DebugInfo(
    #     period=1, should_display_day=True, should_display_action=True, should_display_grid=True, should_display_traits=True), max_days=1)

    detailed = False
    is_main_simulator = True

    simulator = main_simulator if is_main_simulator else tiny_simulator

    if detailed:
        days_survived, log = simulator.run(mutation_rates, debug_info=DebugInfo(
            period=1, should_display_action=True, should_display_day=True, should_display_population=True, should_display_grid=True, should_display_traits=True), max_days=1000)
    else:
        #days_survived, log = simulator.run(mutation_rates, debug_info=DebugInfo(
         #   should_display_day=True, should_display_population=True), max_days=1000)
        days_survived, log = simulator.run(mutation_rates, debug_info=DebugInfo(should_display_population=True), max_days=1000)

    print(log[-1])

    # for log_item in log:
    #     print(log_item)
    print('days', days_survived)