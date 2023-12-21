from simulator import MainSimulator

if __name__ == "__main__":

    main_simulator = MainSimulator()

    mutation_rates = {
        "size": 1,
        "speed": 1,
        "vision": 1,
        "aggression": 1
    }

    log = main_simulator.run(mutation_rates)
