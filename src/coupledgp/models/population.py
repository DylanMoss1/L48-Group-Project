import numpy as np
from GPy.kern import Linear, RBF
from GPy.likelihoods import Gaussian
from GPy.core.gp import GP


class PopulationModel(GP):
    """
    Univariate Gaussian Process model for modelling population changes in the simulation.

    The trained model takes as input:
     - the current temperature
     - the current population
     - the current traits (mean values for size, speed, vision, and aggression)
    and outputs:
     - the next day population

    Args:
        X (np.array): array of input observations of the form (n_samples x n_inputs)
        Y (np.array): array of output observations of the form (n_samples x 1)
    """

    def __init__(self, X: np.array, Y: np.array):
        # current_population_kernel = Linear(1, active_dims=[1])
        # remaining_inputs_kernel = RBF(
        #     X.shape[1] - 1,
        #     lengthscale=0.1,
        #     variance=1,
        #     active_dims=[i for i in range(X.shape[1]) if i != 1],
        # )

        # kernel = current_population_kernel + remaining_inputs_kernel
        kernel = RBF(X.shape[1])

        likelihood = Gaussian(variance=1.0)

        super().__init__(X, Y, kernel, likelihood, name="Population Model")
