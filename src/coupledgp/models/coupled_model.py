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

from utils import (
    format_data_for_drift_model,
    format_data_for_population_model,
)
from utils import emulator_inputs as ei
from models.genetic_drift import GeneticDriftModel
from models.population import PopulationModel


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
     - the initial day
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
            drift_gpy, drift_gpy.num_outputs
        )
        pop_gpy = PopulationModel(x_pop, y_pop)
        self.pop_emukit = GPyModelWrapper(pop_gpy)

        # define input parameter spaces for sensitivity analysis
        # random start day within a year
        day = DiscreteParameter("day", range(1, 366))
        # 100 x 100 grid (10 - 30% populated)
        population = DiscreteParameter("population", range(1000, 3000))
        # need to tweak these limits possibly
        size = ContinuousParameter("size", 0.5, 2.5)
        speed = ContinuousParameter("speed", 0.5, 2.5)
        vision = ContinuousParameter("vision", 0.5, 2.5)
        aggression = ContinuousParameter("aggression", 0.5, 2.5)
        m_size = ContinuousParameter("m_size", 0, 5)
        m_speed = ContinuousParameter("m_speed", 0, 5)
        m_vision = ContinuousParameter("m_vision", 0, 5)
        m_aggression = ContinuousParameter("m_aggression", 0, 5)
        # between a day and a month
        coupling = DiscreteParameter("coupling", [1, 7, 30])

        self.space = ParameterSpace(
            [
                population,
                m_size,
                m_speed,
                m_vision,
                m_aggression,
                coupling,
            ]
        )
        self.drift_space = ParameterSpace(
            [
                day,
                population,
                size,
                speed,
                vision,
                aggression,
                m_size,
                m_speed,
                m_vision,
                m_aggression,
            ]
        )
        self.pop_space = ParameterSpace(
            [day, population, size, speed, vision, aggression]
        )

    def predict(self, X: np.ndarray) -> Tuple[int, List[EmulatorLogItem]]:
        """
        Predicts the number of days until extinction by iteratively alternating between running the genetic drift emulator and the population emulator for a fixed number of time steps. The inputs of both emulators can either feed back into itself or feed forward into the each other.

        Args:
            X (np.ndarray): an array of inputs to predict in the form (n_samples x n_inputs) where the inputs are (population, m_size, m_speed, m_vision, m_aggression, coupling)

        Returns:
            int: the number of days until extinction
            (not implemented yet) List[EmulatorLogItem]: a log of the day/population/traits at each emulated timestep
        """
        # split the input array into three based on coupling values and remove coupling from the inputs
        day_inputs = X[X[:, -1] == 1, :-1]
        week_inputs = X[X[:, -1] == 7, :-1]
        month_inputs = X[X[:, -1] == 30, :-1]
        # store mutation rates for input conversions
        day_mutation_rates = day_inputs[:, 1:5]
        week_mutation_rates = week_inputs[:, 1:5]
        month_mutation_rates = month_inputs[:, 1:5]
        # lists to store completed runs (population <= 0)
        completed_days = []
        completed_weeks = []
        completed_months = []

        # start emulation with changes in population
        day_emulator = "population"
        week_emulator = "population"
        month_emulator = "population"

        day = 1
        day_extinct = False
        week_extinct = False
        month_extinct = False
        # start emulation
        while not day_extinct and not week_extinct and not month_extinct:
            if not day_extinct:  # run day
                day_output, day_inputs, day_emulator = self.run_emulation_step(
                    day_inputs, day_mutation_rates, day_emulator, day, 1
                )
            if not week_extinct:  # swap week
                (
                    week_output,
                    week_inputs,
                    week_emulator,
                ) = self.run_emulation_step(
                    week_inputs, week_mutation_rates, week_emulator, day, 7
                )
            if not month_extinct:  # swap month
                (
                    month_output,
                    month_inputs,
                    month_emulator,
                ) = self.run_emulation_step(
                    month_inputs, month_mutation_rates, month_emulator, day, 30
                )

            # check if population <= 0 and split inputs
            completed_days.append((day, len(day_inputs[:, 1] <= 0)))
            completed_weeks.append((day, len(week_inputs[:, 1] <= 0)))
            completed_months.append((day, len(month_inputs[:, 1] <= 0)))
            day_filter = day_inputs[:, 1] > 0
            week_filter = week_inputs[:, 1] > 0
            month_filter = month_inputs[:, 1] > 0
            day_inputs = day_inputs[day_filter, :]
            day_mutation_rates = day_mutation_rates[day_filter, :]
            week_inputs = week_inputs[week_filter, :]
            week_mutation_rates = week_mutation_rates[week_filter, :]
            month_inputs = month_inputs[month_filter, :]
            month_mutation_rates = month_mutation_rates[month_filter, :]

            # check if all inputs have finished (extinct)
            day_extinct = len(day_inputs) == 0
            week_extinct = len(week_inputs) == 0
            month_extinct = len(month_inputs) == 0

        return (
            completed_days,
            completed_weeks,
            completed_months,
        )  # temporary return, probably will write code to format this better in the future

    def run_emulation_step(
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
            Tuple[np.ndarray, np.ndarray, str]: tuple of (current output, next input, next emulator name)
        """
        if emulator_name == "population":
            emulator = self.pop_emukit
            swap_name = "drift"
            input_function = ei.population_to_population
            swap_function = ei.population_to_drift
        else:
            emulator = self.drift_emukit
            swap_name = "population"
            input_function = ei.drift_to_drift
            swap_function = ei.drift_to_population

        Y = emulator.predict(X)[0]
        if (day % coupling) == 0:
            next_input = swap_function(X, Y, day, mutation_rates)
            return Y, next_input, swap_name
        else:
            next_input = input_function(X, Y, day, mutation_rates)
            return Y, next_input, emulator_name

    def drift_sensitivity_analysis(
        self, graph_results: bool = True, save_plot: bool = False
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

            if save_plot:
                fig.savefig("/drift_sensitivity.png")
        return main_effects, total_effects

    def population_sensitivity_analysis(
        self, graph_results: bool = True, save_plot: bool = False
    ):
        sensitivity = MonteCarloSensitivity(self.pop_emukit, self.pop_space)
        main_effects, total_effects, _ = sensitivity.compute_effects(
            num_monte_carlo_points=10000
        )

        if graph_results:
            fig, axes = plt.subplots(1, 2)
            sns.barplot(main_effects, ax=axes[0])
            sns.barplot(total_effects, ax=axes[1])

            if save_plot:
                fig.savefig("/population_sensitivity.png")
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
        return self.X, self.drift_emukit.X, self.pop_emukit.X

    @property
    def Y(self) -> np.ndarray:
        return self.Y, self.drift_emukit.Y, self.pop_emukit.Y

    @property
    def DriftModel(self) -> IModel:
        return self.drift_emukit

    @property
    def PopulationModel(self) -> IModel:
        return self.pop_emukit

    def save_models(self, drift_file, population_file):
        np.save(drift_file, self.drift_emukit.gpy_model.param_array)
        np.save(population_file, self.pop_emukit.model.param_array)

    def load_models(self, drift_file, population_file):
        self.drift_emukit.gpy_model.update_model(False)
        self.drift_emukit.gpy_model[:] = np.load(drift_file)
        self.drift_emukit.gpy_model.update_model(True)
        self.pop_emukit.model.update_model(False)
        self.pop_emukit.model[:] = np.load(population_file)
        self.pop_emukit.model.update_model(True)
