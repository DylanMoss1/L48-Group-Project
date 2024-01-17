from dataclasses import dataclass
from datetime import datetime
import os
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from GPy.util.multioutput import build_XY
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.sensitivity.monte_carlo import (
    ModelFreeMonteCarloSensitivity,
    MonteCarloSensitivity,
)
from emukit.core.interfaces import IModel
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.model_wrappers import GPyModelWrapper, GPyMultiOutputWrapper

from ..utils import *
from ..models.genetic_drift import GeneticDriftModel
from ..models.population import PopulationModel
from constants import NUM_INITIAL_SPECIES_FRACTION


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

        # for plotting label purposes
        self.trait_names = ["size", "speed", "vision", "aggression"]

    def predict(
        self, X: np.ndarray, n_samples: int = 1000, max_iters: int = None
    ) -> np.ndarray:
        """
        Predicts the number of days until extinction by iteratively alternating between running the genetic drift emulator and the population emulator for a fixed number of time steps. The inputs of both emulators can either feed back into itself or feed forward into the each other.

        Args:
            X (np.ndarray): an array of inputs to predict in the form (n_samples x n_inputs) where the inputs are (population, m_size, m_speed, m_vision, m_aggression, coupling)
            n_samples (int): number of iterations to run and average to propogate the variance
            max_iters (int, optional): sets a hard limit on the number of iterations. Any unfinished simulations will have max_iters as an output. Defaults to None.

        Returns:
            np.ndarray: the number of days until extinction for each input
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
        final_means = np.zeros((X.shape[0], 1))
        final_vars = np.zeros((X.shape[0], 1))

        day = 1
        # initialize arrays for execution loop
        is_extinct = [False for _ in coupling_times]
        # convert coupled inputs to population inputs, following the simulator's method of initializing traits (random between 0 and 1)
        split_temp_inputs = [
            coupled_to_population(
                x, day, np.random.rand(x.shape[0], len(self.trait_names))
            )
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
                        n_samples,
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
                    dead_indices = population[population[:, 1] <= 0, 0].astype(
                        int
                    )
                    if len(dead_indices) > 0:
                        final_means[dead_indices] = day

                    # check if this group has gone extinct
                    is_extinct[i] = len(split_temp_inputs[i]) == 0

            day += 1
            if max_iters is not None and day > max_iters:
                final_means[final_means == 0] = max_iters
                break

        return final_means

    def _run_emulation_step(
        self,
        X: np.ndarray,
        mutation_rates: np.ndarray,
        emulator_name: str,
        day: int,
        coupling: int,
        n_samples: int,
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

        sampled_X = np.tile(X, (n_samples, 1))
        sampled_Y_mean, sampled_Y_var = emulator.predict(
            X[:, 1:]
        )  # remove index from input
        Y = emulator.predict(X[:, 1:])[0]
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

    def plot_training(self, loop_state, plot_type: str):
        mr_range = np.linspace(0, 1, 100)
        trait_range = np.linspace(0, 1, 100)
        fig, axes = plt.subplots(4, 1, figsize=(20, 20))
        if plot_type == "coupled":
            for i in range(4):
                X = np.hstack(
                    [
                        np.full((len(mr_range), 1), 500),
                        np.zeros((len(mr_range), 4)),
                        np.ones((len(mr_range), 1))
                    ]
                )
                X[:, 1 + i] = mr_range
                mean = self.predict(X)
                axes[i].plot(mr_range, mean.flatten())
                axes[i].set(
                    xlabel="mr", ylabel="days", title=f"{self.trait_names[i]}"
                )
        elif plot_type == "drift":
            for i in range(4):
                X = np.hstack(
                    [
                        np.full((len(trait_range), 1), 10),
                        np.full((len(mr_range), 1), 500),
                        np.full((len(mr_range), NUM_TRAITS), 0.5),
                        np.zeros((len(mr_range), 4)),
                        np.full((len(mr_range), 1), i),
                    ]
                )
                X[:, 10 + i] = mr_range
                mean, var = self.drift_emukit.predict(X)
                mean = mean.flatten()
                var = var.flatten()
                axes[i].plot(mr_range, mean)
                axes[i].fill_between(
                    mr_range,
                    mean + np.sqrt(var),
                    mean - np.sqrt(var),
                    alpha=0.6,
                )
                axes[i].fill_between(
                    mr_range,
                    mean + 2 * np.sqrt(var),
                    mean - 2 * np.sqrt(var),
                    alpha=0.3,
                )
                axes[i].set(
                    xlabel="mr", ylabel="trait", title=f"{self.trait_names[i]}"
                )
        elif plot_type == "population":
            for i in range(4):
                X = np.hstack(
                    [
                        np.full((len(trait_range), 1), 10),
                        np.full((len(mr_range), 1), 500),
                        np.full((len(mr_range), NUM_TRAITS), 0.5),
                    ]
                )
                X[:, 2 + i] = trait_range
                mean, var = self.pop_emukit.predict(X)
                mean = mean.flatten()
                var = var.flatten()
                axes[i].plot(trait_range, mean)
                axes[i].fill_between(
                    trait_range,
                    mean + np.sqrt(var),
                    mean - np.sqrt(var),
                    alpha=0.6,
                )
                axes[i].fill_between(
                    trait_range,
                    mean + 2 * np.sqrt(var),
                    mean - 2 * np.sqrt(var),
                    alpha=0.3,
                )
                axes[i].set(
                    xlabel="trait",
                    ylabel="population",
                    title=f"{self.trait_names[i]}",
                )
        plt.plot()
        fig.savefig(
            f"./src/coupledgp/tests/plots/{plot_type}_training_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
        )

    def plot_drift_model(self, show_plot: bool = True, save_plot: bool = True):
        """
        Plots drift model fit for trait value vs. trait mutation rate for each trait across several temperatures (-5, 10 [optimal], 25), keeping all initial traits (0.5), all other mutation rates (0.1), and population (500) constant.
        """
        fixed_values = {
            "population": NUM_INITIAL_SPECIES_FRACTION * (50 * 50),
            "size": 0.5,
            "speed": 0.5,
            "vision": 0.5,
            "aggression": 0.5,
            "size_mr": 0.1,
            "speed_mr": 0.1,
            "vision_mr": 0.1,
            "aggression_mr": 0.1,
        }
        temperatures = [-5, 10, 25]
        fig, axes = plt.subplots(
            len(self.trait_names), 1, sharex="all", figsize=(20, 20)
        )
        for i in range(len(self.trait_names)):
            for t in temperatures:
                fixed_inputs = [
                    (j + 1, fixed_values[k])
                    for j, k in enumerate(fixed_values)
                    if (j + 1) != (6 + i)
                ]
                fixed_inputs.append((10, i))  # add output index
                fixed_inputs.append((0, t))  # add temperature
                self.drift_emukit.gpy_model.plot(
                    ax=axes[i],
                    fixed_inputs=fixed_inputs,
                    visible_dims=[6 + i],
                    label=f"temp: {t}",
                )
            axes[i].set_xlabel(f"{self.trait_names[i]} mr")
            axes[i].set_ylabel(self.trait_names[i])
            axes[i].set_xlim(0, 1)
            axes[i].set_ylim(0, 1)

        if save_plot:
            fig.savefig(
                f"./src/coupledgp/tests/plots/drift_plot_mr_w_rest_fixed_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
            )
        if show_plot:
            plt.show()

        # plotting how different trait mutation rates affect a trait's evolution

    def plot_population_model(
        self, show_plot: bool = True, save_plot: bool = True
    ):
        """
        Plots population model fit for population vs. trait value for each trait across several temperatures (-5, 10 [optimal], 25), keeping all other traits fixed at 0.5 and population fixed at 500.
        """
        fixed_values = {
            "population": NUM_INITIAL_SPECIES_FRACTION * (50 * 50),
            "size": 0.5,
            "speed": 0.5,
            "vision": 0.5,
            "aggression": 0.5,
        }
        temperatures = [-5, 10, 25]
        fig, axes = plt.subplots(
            len(self.trait_names), 1, sharex="all", figsize=(20, 20)
        )
        for i in range(len(self.trait_names)):
            for t in temperatures:
                fixed_inputs = [
                    (j + 1, fixed_values[k])
                    for j, k in enumerate(fixed_values)
                    if (j + 1) != 2 + i
                ]
                fixed_inputs.append((0, t))  # add temperature
                self.pop_emukit.model.plot(
                    ax=axes[i],
                    fixed_inputs=fixed_inputs,
                    visible_dims=[2 + i],
                    label=f"temp: {t}",
                )
            axes[i].set_xlabel(self.trait_names[i])
            axes[i].set_ylabel("population")
            axes[i].set_xlim(0, 1)

        if save_plot:
            fig.savefig(
                f"./src/coupledgp/tests/plots/population_plot_trait_w_rest_fixed_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
            )
        if show_plot:
            plt.show()

    def plot_coupled_model(
        self, n_samples: int, show_plot: bool = True, save_plot: bool = True
    ):
        """
        Plots time until extinction vs. mutation rate for each mutation rate, keeping other mutation rates fixed at 0.1 and population fixed to 500 (default).

        Args:
            n_samples (int): Number of samples to use for each plot.
        """
        design = LatinDesign(coupled_space)
        input_template = design.get_samples(n_samples)
        # fix population to simulation initial population and coupling to 1
        input_template[:, 0] = NUM_INITIAL_SPECIES_FRACTION * (50 * 50)
        input_template[:, -1] = 1

        fig, axes = plt.subplots(4, 1, sharex="all", figsize=(20, 20))
        for i in range(len(self.trait_names)):
            modified_input = input_template.copy()
            modified_input[:, 1:5] = 0.1
            modified_input[:, 1 + i] = input_template[:, 1 + i]

            # run emulator for 1000 steps or until finished
            outputs = self.predict(modified_input, max_iters=1000)
            x = modified_input[:, 1 + i]  # mutation rate
            y = outputs[:, 0]
            axes[i].scatter(x, y)
            axes[i].set_xlabel(f"{self.trait_names[i]} mr")
            axes[i].set_ylabel("Days survived")

        if save_plot:
            fig.savefig(
                f"./src/coupledgp/tests/plots/coupled_plot_mr_w_rest_fixed_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
            )
        if show_plot:
            plt.show()

    def drift_sensitivity_analysis(
        self, show_plot: bool = True, save_plot: bool = True
    ):
        # plot sensitivities per output
        fig, axes = plt.subplots(len(self.trait_names), 2, figsize=(20, 20))
        for i in range(len(self.trait_names)):
            sensitivity = ModelFreeMonteCarloSensitivity(
                lambda x: self.drift_emukit.predict(
                    np.append(x[:, :-1], np.full((x.shape[0], 1), i), axis=1)
                )[0],
                self.drift_space,
            )
            main_effects, total_effects, _ = sensitivity.compute_effects(
                num_monte_carlo_points=10000
            )

            sns.barplot(main_effects, ax=axes[i][0])
            sns.barplot(total_effects, ax=axes[i][1])
            axes[i][0].set_title(f"Main Effects ({self.trait_names[i]})")
            axes[i][1].set_title(f"Total Effects ({self.trait_names[i]})")

        if save_plot:
            fig.savefig(
                f"./src/coupledgp/tests/plots/drift_sensitivity_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
            )
        if show_plot:
            plt.show()

    def population_sensitivity_analysis(
        self, show_plot: bool = True, save_plot: bool = True
    ):
        sensitivity = MonteCarloSensitivity(self.pop_emukit, self.pop_space)
        main_effects, total_effects, _ = sensitivity.compute_effects(
            num_monte_carlo_points=10000
        )

        fig, axes = plt.subplots(1, 2)
        sns.barplot(main_effects, ax=axes[0])
        sns.barplot(total_effects, ax=axes[1])
        axes[0].set_title("Main Effects")
        axes[1].set_title("Total Effects")

        if save_plot:
            fig.savefig(
                f"./src/coupledgp/tests/plots/population_sensitivity_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
            )
        if show_plot:
            plt.show()

    def coupled_sensitivity_analysis(
        self, show_plot: bool = True, save_plot: bool = True
    ):
        sensitivity = ModelFreeMonteCarloSensitivity(
            lambda x: self.predict(x), coupled_space
        )
        main_effects, total_effects, _ = sensitivity.compute_effects(
            num_monte_carlo_points=10000
        )

        fig, axes = plt.subplots(1, 2)
        sns.barplot(main_effects, ax=axes[0])
        sns.barplot(total_effects, ax=axes[1])
        axes[0].set_title("Main Effects")
        axes[1].set_title("Total Effects")

        if save_plot:
            fig.savefig(
                f"./src/coupledgp/tests/plots/coupled_sensitivity_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
            )
        if show_plot:
            plt.show()

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X = X
        self.Y = Y
        x_list, y_list = format_data_for_drift_model(X, Y)
        x_drift, y_drift, _ = build_XY(x_list, y_list)
        x_pop, y_pop = format_data_for_population_model(X, Y)

        self.drift_emukit.set_data(x_drift, y_drift)
        self.pop_emukit.set_data(x_pop, y_pop)

    def optimize(self, verbose: bool = False) -> None:
        self.drift_emukit.gpy_model.kern.fix()
        self.pop_emukit.model.kern.fix()
        self.drift_emukit.optimize()
        self.pop_emukit.optimize()
        self.drift_emukit.gpy_model.kern.constrain_positive()
        self.pop_emukit.model.kern.constrain_positive()


    def compare_with_simulator(
        self,
        simulator_input_path="dataset_test/mutation_rates.npy",
        simulator_output_path="dataset_test/simulated_years_of_survival.npy",
    ):
        """Inputs in (n_samples, 4), outputs in (n_samples, 1)"""
        inputs = np.load(simulator_input_path)
        outputs = np.load(simulator_output_path)
        initial_population = 500
        model_input = np.hstack(
            [
                np.full((inputs.shape[0], 1), initial_population),
                inputs,
                np.ones((inputs.shape[0], 1)),
            ]
        )
        print(model_input)
        model_output = self.predict(model_input)
        se = (model_output - outputs) ** 2

        mse = np.mean(se)
        return mse

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
