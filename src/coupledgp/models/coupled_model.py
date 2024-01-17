from dataclasses import dataclass
from datetime import datetime
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
from emukit.bayesian_optimization.loops import (
    BayesianOptimizationLoop,
)
from emukit.bayesian_optimization.acquisitions import (
    ExpectedImprovement,
    ProbabilityOfImprovement,
    NegativeLowerConfidenceBound,
)
from emukit.benchmarking.loop_benchmarking.benchmarker import Benchmarker
from emukit.benchmarking.loop_benchmarking.metrics import (
    MeanSquaredErrorMetric,
)
from emukit.benchmarking.loop_benchmarking.benchmark_plot import BenchmarkPlot
from emukit.model_wrappers import GPyModelWrapper, GPyMultiOutputWrapper

from ..utils import *
from ..data import SimulateCoupled, SimulateDrift, SimulatePopulation
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
        self.X = X
        self.Y = Y
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
        self,
        X: np.ndarray,
        n_samples: int = 1000,
        max_iters: int = None,
        seed: int = 0,
    ) -> np.ndarray:
        """
        Predicts the number of days until extinction by iteratively alternating between running the genetic drift emulator and the population emulator for a fixed number of time steps. The inputs of both emulators feed forward into each other.

        Args:
            X (np.ndarray): an array of inputs to predict in the form (n_inputs x [population, m_size, m_speed, m_vision, m_aggression])
            n_samples (int): number of iterations to run and average to propogate the variance
            max_iters (int, optional): sets a hard limit on the number of iterations. Any unfinished simulations will have max_iters as an output. Defaults to None.
            seed (int): sets the random seed

        Returns:
            Tuple[np.ndarray, np.ndarray]: the mean and variance for the number of days until extinction for each input
        """
        # initializes random generator
        rng = np.random.default_rng(seed)
        # adds input indices at the beginning for distinguishing inputs
        X = np.insert(X, [0], np.arange(X.shape[0])[:, None], axis=1)

        # initialize output arrays to 0 (final form is (n_inputs x n_samples) array)
        final_extinction_days = np.zeros((X.shape[0], n_samples))

        # start emulation
        for x_input in X:
            print(f"Emulating input: {x_input}")
            # store mutation rates for input conversions
            mutation_rates = x_input[2:]
            ## prepare input for emulator
            # repeat n_sample times to represent drawing from a trait uniform distribution n_sample times (different trait values)
            temp_x = np.tile(x_input, (n_samples, 1))
            traits = rng.uniform(0, 1, (n_samples, NUM_TRAITS))
            # convert coupled inputs to population inputs
            temp_x = coupled_to_population(temp_x, 1, traits)
            # add indices to identify emulation runs
            temp_x = np.insert(
                temp_x, [1], np.arange(temp_x.shape[0])[:, None], axis=1
            )
            # initialize execution loop
            day = 1
            is_all_extinct = False
            while not is_all_extinct:
                # run emulation step for non-extinct inputs
                temp_x = self._run_emulation_step(
                    temp_x, mutation_rates, day, rng
                )

                # do population checks (n_remaining_samples x 12)
                dead_indices = np.where(temp_x[:, 3] <= 0)[0]
                alive_indices = np.where(temp_x[:, 3] > 0)[0]
                # get population values (sample_index)
                for dead_index in dead_indices:
                    input_index = int(temp_x[dead_index, 0])
                    sample_index = int(temp_x[dead_index, 1])
                    print(input_index, sample_index)
                    if (
                        final_extinction_days[input_index, sample_index] == 0
                    ):  # i.e., new extinction
                        final_extinction_days[input_index, sample_index] = day
                # remove any fully extinct samples
                temp_x = temp_x[alive_indices, :]
                # book-keeping at the end
                day += 1
                is_all_extinct = len(temp_x) == 0
                if max_iters is not None and day > max_iters:
                    final_extinction_days[
                        final_extinction_days == 0
                    ] = max_iters
                    break
        final_means = np.mean(final_extinction_days, axis=1)
        final_vars = np.var(final_extinction_days, axis=1)
        return final_means, final_vars

    def _run_emulation_step(
        self,
        pop_X: np.ndarray,
        mutation_rates: np.ndarray,
        day: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Runs a single day of emulation (population -> drift ->)

        Args:
            pop_X (np.ndarray): input of the current emulation step (population emulator).
                This is in the form (n_samples x [input_index, sample_index, temperature, population, size, speed, vision, aggression, size_var, speed_var, vision_var, aggression_var]).
            mutation_rates (np.ndarray): mutation rate information for swapping to drift inputs
                This is in the form (4)

        Returns:
            np.ndarray: next (population) input
        """
        sampled_Y_mean, sampled_Y_var = self.pop_emukit.predict(
            pop_X[:, 2:]
        )  # remove index from input
        Y_mean = np.mean(sampled_Y_mean)
        Y_var = np.sum((sampled_Y_var / len(sampled_Y_var)) ** 2)

        drift_X = population_to_drift(
            pop_X, Y_mean, Y_var, day, rng, mutation_rates
        )
        sampled_Y_mean, sampled_Y_var = self.pop_emukit.predict(drift_X[:, 2:])
        Y_mean = np.mean(
            np.hstack(np.vsplit(sampled_Y_mean, NUM_TRAITS)), axis=0
        ).reshape((NUM_TRAITS, 1))
        Y_var = np.sum(
            (
                np.hstack(np.vsplit(sampled_Y_var, NUM_TRAITS))
                / len(sampled_Y_var)
                / NUM_TRAITS
            )
            ** 2,
            axis=0,
        ).reshape((NUM_TRAITS, 1))

        pop_X = drift_to_population(drift_X, Y_mean, Y_var, day, rng)
        return pop_X

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X = X
        self.Y = Y
        x_list, y_list = format_data_for_drift_model(X, Y)
        x_drift, y_drift, _ = build_XY(x_list, y_list)
        x_pop, y_pop = format_data_for_population_model(X, Y)

        self.drift_emukit.set_data(x_drift, y_drift)
        self.pop_emukit.set_data(x_pop, y_pop)

    def optimize(self, verbose: bool = False) -> None:
        self.drift_emukit.optimize(verbose=verbose)
        self.pop_emukit.optimize(verbose=verbose)

    def get_bayes_opt_loop(self, acquisition, update_interval):
        bo_loop = BayesianOptimizationLoop(
            coupled_space, self, acquisition, update_interval, 1
        )
        return bo_loop

    def get_drift_bayes_opt_loop(self, acquisition, update_interval):
        bo_loop = BayesianOptimizationLoop(
            drift_space, self.drift_emukit, acquisition, update_interval, 1
        )
        return bo_loop

    def get_population_bayes_opt_loop(self, acquisition, update_interval):
        bo_loop = BayesianOptimizationLoop(
            population_space, self.pop_emukit, acquisition, update_interval, 1
        )
        return bo_loop

    def train(
        self,
        acquisition,
        n_iterations: int = 10,
        update_interval: int = 5,
        train_components: bool = False,
    ):
        if train_components:
            drift_bo_loop = self.get_drift_bayes_opt_loop(
                acquisition, update_interval
            )
            pop_bo_loop = self.get_population_bayes_opt_loop(
                acquisition, update_interval
            )
            drift_bo_loop.run_loop(SimulateDrift(), n_iterations)
            pop_bo_loop.run_loop(SimulatePopulation(), n_iterations)
            drift_results = drift_bo_loop.loop_state
            pop_results = pop_bo_loop.loop_state
            self.plot_training(drift_results)
            self.plot_training(pop_results)
            return drift_results, pop_results
        else:
            bo_loop = self.get_bayes_opt_loop(acquisition, update_interval)
            bo_loop.run_loop(SimulateCoupled(), n_iterations)
            results = bo_loop.loop_state
            self.plot_training(results)
            return results

    def train_comparison(
        self,
        n_iterations: int = 10,
        n_initial_data: int = 1,
        update_interval: int = 5,
    ):
        inputs = np.load("./src/dataset_test/mutation_rates.npy")
        outputs = np.load("./src/dataset_test/simulated_years_of_survival.npy")
        initial_population = 500
        model_input = np.hstack(
            [
                np.full((inputs.shape[0], 1), initial_population),
                inputs,
                np.ones((inputs.shape[0], 1)),
            ]
        )
        loops = [
            (
                "Expected Improvement",
                lambda loop_state: self.get_bayes_opt_loop(
                    ExpectedImprovement(
                        CoupledGPModel(loop_state.X, loop_state.Y)
                    ),
                    update_interval,
                ),
            ),
            (
                "Probability of Improvement",
                lambda loop_state: self.get_bayes_opt_loop(
                    ProbabilityOfImprovement(
                        CoupledGPModel(loop_state.X, loop_state.Y)
                    ),
                    update_interval,
                ),
            ),
            (
                "Negative LCB",
                lambda loop_state: self.get_bayes_opt_loop(
                    NegativeLowerConfidenceBound(
                        CoupledGPModel(loop_state.X, loop_state.Y)
                    ),
                    update_interval,
                ),
            ),
        ]
        metrics = [MeanSquaredErrorMetric(model_input, outputs)]
        benchmarker = Benchmarker(
            loops, SimulateCoupled(), coupled_space, metrics=metrics
        )
        benchmark_results = benchmarker.run_benchmark(
            n_iterations=n_iterations,
            n_initial_data=n_initial_data,
            n_repeats=10,
        )
        plots = BenchmarkPlot(
            benchmark_results,
            loop_colors=["m", "c", "g"],
            loop_line_styles=["-", "--", "-."],
            metrics_to_plot=["mean_squared_error"],
        )
        plots.make_plot()
        plots.save_plot(
            f"./src/coupledgp/tests/plots/acquisition_comparison_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
        )

    def plot_training(self, loop_state, plot_type: str):
        mr_range = np.linspace(0, MAX_MUTATION_RATE, 100)
        trait_range = np.linspace(0, MAX_TRAIT_VALUE, 100)
        fig, axes = plt.subplots(4, 1, figsize=(20, 20))
        if plot_type == "coupled":
            for i in range(4):
                X = np.hstack(
                    [
                        np.full((len(mr_range), 1), 500),
                        np.zeros((len(mr_range), 4)),
                    ]
                )
                X[:, 1 + i] = mr_range
                mean, var = self.predict(X)
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

    def compare_with_simulator(
        self,
        simulator_input_path="./src/dataset_test/mutation_rates.npy",
        simulator_output_path="./src/dataset_test/simulated_years_of_survival.npy",
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
        model_output, _ = self.predict(model_input)
        se = (model_output - outputs) ** 2

        mse = np.mean(se)
        return mse

    @property
    def X(self) -> np.ndarray:
        return self.X

    @property
    def Y(self) -> np.ndarray:
        return self.Y

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
            mean, var = self.predict(modified_input, max_iters=1000)
            x = modified_input[:, 1 + i]  # mutation rate
            y_mean = mean[:, 0]
            y_std = np.sqrt(var)[:, 0]
            axes[i].plot(x, y_mean)
            axes[i].fill_between(x, y_mean + y_std, y_mean - y_std, alpha=0.6)
            axes[i].fill_between(
                x, y_mean + 2 * y_std, y_mean - 2 * y_std, alpha=0.4
            )
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
        sensitivity = MonteCarloSensitivity(self, coupled_space)
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
