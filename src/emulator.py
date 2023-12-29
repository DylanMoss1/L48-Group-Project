from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper
from emukit.core.acquisition import Acquisition
from emukit.core.parameter_space import ParameterSpace
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.experimental_design import ExperimentalDesignLoop
from emukit.sensitivity.monte_carlo import MonteCarloSensitivity

NOISE_VAR = 1.0
INITIAL_SAMPLE = 20
BATCH_SIZE = 5
MAX_ITERS = 10

class BaseEmulator(ABC):
    def __init__(self, space: ParameterSpace, acquisition: Acquisition):
        """Initializes the GP model, EmuKit wrapper, and initial sampling points.

        Args:
            space (ParameterSpace): Describes the input variables to the model and any constraints.
            acquisition (Acquisition): Describes the acquisition function to use for optimization.
        """
        # initial sample of points
        design = LatinDesign(space)
        X = design.get_samples(INITIAL_SAMPLE)
        Y = self.run_simulator(X)
        
        # initialize model
        gpy_model = GPRegression(X, Y, NOISE_VAR)
        self.model = GPyModelWrapper(gpy_model)
        self.parameter_space = space
        self.acquisition_function = acquisition(model = self.model)
    
    @abstractmethod
    def run_simulator(self, params: np.ndarray) -> np.ndarray:
        """Function to generate data to train the emulator.

        Args:
            params (np.ndarray): An array of (point_count x space_dim). The inputs to run the function (simulator) at. Determined by the sampling process and parameter space.

        Returns:
            np.ndarray: The outputs of the function (in this case, the simulator).
        """
        return NotImplemented
    
    def train(self) -> None:
        """Runs the experimental design loop to train the emulator.
        
        Updates model hyper-parameters using MLE every 'batch_size' iterations.
        Uses the acquisition function to determine which points to sample next.
        """
        expdesign_loop = ExperimentalDesignLoop(model = self.model,
                                                space = self.parameter_space,
                                                acquisition = self.acquisition_function,
                                                batch_size = BATCH_SIZE)
        expdesign_loop.run_loop(self.run_simulator, MAX_ITERS)
    
    def sensitivity_analysis(self, save_results: bool = False, save_location: str = None) -> None:
        """Performs sensitivity analysis using Monte Carlo integration methods with emulator sampling.
        
        Graphs the first-order sobel indices and total effects for each input variable.

        Args:
            save_results (bool, optional): Save the resulting bar graph. Defaults to False.
            save_location (str, optional): Save location. Defaults to None.
        """
        senstivity = MonteCarloSensitivity(model = self.model, input_domain = self.parameter_space)
        main_effects, total_effects, _ = senstivity.compute_effects(num_monte_carlo_points = 10000)
        
        fig, axes = plt.subplots(1, 2)
        sns.barplot(main_effects, ax=axes[0])
        sns.barplot(total_effects, ax=axes[1])
        
        if save_results:
            if save_location is None:
                save_location = "/sensitivity.png"
            fig.savefig(save_location)

class GeneticDriftModel(BaseEmulator):
    pass

class PopulationModel(BaseEmulator):
    pass