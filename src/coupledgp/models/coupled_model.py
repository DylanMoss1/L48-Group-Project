from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from GPy.util.multioutput import build_XY
from emukit.core import ParameterSpace, DiscreteParameter, ContinuousParameter
from emukit.sensitivity.monte_carlo import MonteCarloSensitivity
from emukit.core.interfaces import IModel
from emukit.model_wrappers import GPyModelWrapper, GPyMultiOutputWrapper

from ..utils import *
from ..models.genetic_drift import GeneticDriftModel
from ..models.population import PopulationModel


@dataclass
class EmulatorLogItem:
    """
    Contains a log entries for each day of the emulation, for analysis and evaluation.

    Attributes
    ----------
    day : int
        The current day of the emulation
    num_species_alive : int
        The population as input into the emulator
    traits_dict : Dict[str, float]
        Contains the mean traits as input into the emulator with keys: size, speed, vision, aggression
    """

    day: int
    num_species_alive: int
    traits_dict: Dict[str, float]


class CoupledGPModel(IModel):
    """
    Wrapper for the coupled model between genetic drift and population.

    The model takes as input:
     - the initial population
     - the mutation rates
     - the coupling period
    and outputs:
     - total days until extinction

    The output is obtained by iteratively switching between running the population model for [coupling period] timesteps, and then feeding the result into running the genetic drift model for [coupling period] timesteps, and so on.
    """

    def __init__(self, X, Y):
        x_drift, y_drift = format_data_for_drift_model(X, Y)
        x_pop, y_pop = format_data_for_population_model(X, Y)
        drift_gpy = GeneticDriftModel(x_drift, y_drift)
        self.drift_emukit = GPyMultiOutputWrapper(
            drift_gpy, drift_gpy.num_outputs, 1
        )
        pop_gpy = PopulationModel(x_pop, y_pop)
        self.pop_emukit = GPyModelWrapper(pop_gpy)

        # define input parameter spaces for sensitivity analysis
        self.drift_space = drift_space
        self.pop_space = population_space

    def predict(self, X: np.ndarray, max_iters: int = None) -> int:
        """
        Predicts the number of days until extinction by iteratively alternating between running the genetic drift emulator and the population emulator for a fixed number of time steps. The inputs of both emulators can either feed back into itself or feed forward into the each other.

        Args:
            X (np.ndarray): an array of inputs to predict in the form (n_samples x n_inputs) where the inputs are (population, m_size, m_speed, m_vision, m_aggression, coupling)
            max_iters (int, optional): sets a hard limit on the number of iterations. Any unfinished simulations will have 0 as an output. Defaults to None.

        Returns:
            int: the number of days until extinction
        """
        # get unique coupling times from inputs
        coupling_times = np.unique(X[:, -1])
        # adds input indices at the beginning for filtering
        X = np.insert(X, [0], np.arange(X.shape[0])[:, None], axis=1)
        # split input array based on coupling values and remove them from inputs
        split_inputs = [X[X[:, -1] == t, :-1] for t in coupling_times]
        # store mutation rates for input conversions
        split_temp_mutation_rates = [x[:, 2:6] for x in split_inputs]
        # start emulation with changes in population
        split_emulators = ["population" for _ in coupling_times]

        # initialize output arrays to 0 (final form is (n_samples x 1) array)
        final_outputs = np.zeros((X.shape[0], 1))

        day = 1
        # initialize arrays for execution loop
        is_extinct = [False for _ in coupling_times]
        # convert coupled inputs to population inputs, following the simulator's method of initializing traits (random between 0 and 1)
        split_temp_inputs = [
            coupled_to_population(x, day, np.random.rand(x.shape[0], 4))
            for x in split_inputs
        ]
        # start emulation
        while not all(is_extinct):
            for i, t in enumerate(coupling_times):
                if not is_extinct[i]:
                    # run emulation step only for inputs with population > 0
                    (
                        population,
                        split_temp_inputs[i],
                        split_temp_mutation_rates[i],
                        split_emulators[i],
                    ) = self._run_emulation_step(
                        split_temp_inputs[i],
                        split_temp_mutation_rates[i],
                        split_emulators[i],
                        day,
                        t,
                    )

                    # filter inputs for population > 0
                    alive_indices = split_temp_inputs[i][:, 2] > 0
                    split_temp_inputs[i] = split_temp_inputs[i][
                        alive_indices, :
                    ]
                    split_temp_mutation_rates[i] = split_temp_mutation_rates[
                        i
                    ][alive_indices, :]

                    # add extinction day to outputs for population <= 0
                    dead_indices = population[population[:, 1] <= 0, 0]
                    if len(dead_indices) > 0:
                        final_outputs[dead_indices] = day

                    # check if this group has gone extinct
                    is_extinct[i] = len(split_temp_inputs[i]) == 0

            day += 1
            if max_iters is not None and day > max_iters:
                break

        return final_outputs

    def _run_emulation_step(
        self,
        X: np.ndarray,
        mutation_rates: np.ndarray,
        emulator_name: str,
        day: int,
        coupling: int,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Runs a single emulation step.

        Args:
            X (np.ndarray): input of the current emulation step
            mutation_rates (np.ndarray): mutation rate information in the case of a swap

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, str]: tuple of (current output, next input, next mutation rate, next emulator name)
        """
        if emulator_name == "population":
            emulator = self.pop_emukit
            swap_name = "drift"
            input_function = population_to_population
            swap_function = population_to_drift
        else:
            emulator = self.drift_emukit
            swap_name = "population"
            input_function = drift_to_drift
            swap_function = drift_to_population
        Y = emulator.predict(X[:, 1:])[0]  # remove index from input
        if (day % coupling) == 0:
            next_input, next_mr, formatted_population = swap_function(
                X, Y, day, mutation_rates
            )
            return formatted_population, next_input, next_mr, swap_name
        else:
            next_input, next_mr, formatted_population = input_function(
                X, Y, day, mutation_rates
            )
            return formatted_population, next_input, next_mr, emulator_name

    def plot_drift_model(self, save_plot: bool = True):
        trait_names = ["size", "speed", "vision", "aggression"]

        # plotting how changing one trait's mutation rate affects that trait's evolution
        fig, axes = plt.subplots(4, 1, sharex="all", figsize=(20, 20))
        for i in range(4):
            self.drift_emukit.gpy_model.plot(
                ax=axes[i],
                fixed_inputs=[(0, 25), (10, i)],
                visible_dims=[6 + i],
            )
            axes[i].set_xlabel("mutation_rate")
            axes[i].set_ylabel(trait_names[i])

        if save_plot:
            fig.savefig("./src/coupledgp/tests/drift_plot_mr_w_25temp.svg")
        plt.show()

        # plotting how different trait mutation rates affect a trait's evolution

    def plot_population_model(self, save_plot: bool = True):
        trait_names = ["size", "speed", "vision", "aggression"]

        # plotting how changing one trait affects the population over temperature
        fig, axes = plt.subplots(4, 1, sharex="all", figsize=(20, 20))
        for i in range(4):
            for v in np.arange(0, 1, 0.1):
                self.pop_emukit.model.plot(
                    ax=axes[i], fixed_inputs=[(0, 25)], visible_dims=[2 + i]
                )
            axes[i].set_xlabel(trait_names[i])
            axes[i].set_ylabel("population")

        if save_plot:
            fig.savefig("./src/coupledgp/tests/population_plot_trait_w_25temp")
        plt.show()

    def drift_sensitivity_analysis(
        self, graph_results: bool = True, save_plot: bool = True
    ):
        sensitivity = MonteCarloSensitivity(
            self.drift_emukit, self.drift_space
        )
        main_effects, total_effects, _ = sensitivity.compute_effects(
            num_monte_carlo_points=10000
        )

        if graph_results:
            fig, axes = plt.subplots(1, 2)
            sns.barplot(main_effects, ax=axes[0])
            sns.barplot(total_effects, ax=axes[1])
            axes[0].set_title("Main Effects")
            axes[1].set_title("Total Effects")

            if save_plot:
                fig.savefig("./src/coupledgp/tests/drift_sensitivity.svg")
            plt.show()
        return main_effects, total_effects

    def population_sensitivity_analysis(
        self, graph_results: bool = True, save_plot: bool = True
    ):
        sensitivity = MonteCarloSensitivity(self.pop_emukit, self.pop_space)
        main_effects, total_effects, _ = sensitivity.compute_effects(
            num_monte_carlo_points=10000
        )

        if graph_results:
            fig, axes = plt.subplots(1, 2)
            sns.barplot(main_effects, ax=axes[0])
            sns.barplot(total_effects, ax=axes[1])
            axes[0].set_title("Main Effects")
            axes[1].set_title("Total Effects")

            if save_plot:
                fig.savefig("./src/coupledgp/tests/population_sensitivity.svg")
            plt.show()
        return main_effects, total_effects

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X = X
        self.Y = Y
        x_list, y_list = format_data_for_drift_model(X, Y)
        x_drift, y_drift, _ = build_XY(x_list, y_list)
        x_pop, y_pop = format_data_for_population_model(X, Y)

        self.drift_emukit.set_data(x_drift, y_drift)
        self.pop_emukit.set_data(x_pop, y_pop)

    def optimize(self, verbose: bool = False) -> None:
        self.drift_emukit.optimize()
        self.pop_emukit.optimize()

    @property
    def X(self) -> np.ndarray:
        return self.drift_emukit.X, self.pop_emukit.X

    @property
    def Y(self) -> np.ndarray:
        return self.drift_emukit.Y, self.pop_emukit.Y

    @property
    def DriftModel(self) -> IModel:
        return self.drift_emukit

    @property
    def PopulationModel(self) -> IModel:
        return self.pop_emukit

    def save_models(self, drift_file, population_file):
        np.save(f"{drift_file}.npy", self.drift_emukit.gpy_model.param_array)
        np.save(f"{population_file}.npy", self.pop_emukit.model.param_array)

    def load_models(self, drift_file, population_file):
        self.drift_emukit.gpy_model.update_model(False)
        self.drift_emukit.gpy_model[:] = np.load(drift_file)
        self.drift_emukit.gpy_model.update_model(True)
        self.pop_emukit.model.update_model(False)
        self.pop_emukit.model[:] = np.load(population_file)
        self.pop_emukit.model.update_model(True)
