from typing import List

import numpy as np
from GPy.util.multioutput import build_XY, build_likelihood, LCM
from GPy.kern import RBF
from GPy.core.gp import GP


class GeneticDriftModel(GP):
    """
    Multi-output Gaussian Process model for modelling genetic drift in the simulation.

    The trained model takes as input:
     - the current temperature
     - the current population
     - the current traits (mean values for size, speed, vision, and aggression)
     - the current mutation rates
    and outputs:
     - the next day traits (mean values for size, speed, vision, and aggression)

    Args:
        X_list (list(np.array)): list of input observations corresponding to each output of the form (n_outputs x (n_samples x n_inputs))
        Y_list (list(np.array)): list of output observations of the form (n_outputs x (n_samples x 1))
    """

    def __init__(self, X_list: List[np.array], Y_list: List[np.array]):
        X, Y, self.output_index = build_XY(X_list, Y_list)
        self.num_outputs = len(Y_list)

        # we model each output function f_d(x) as a linear combination of Q GPs with covariance k_q
        # thus, covariance is the sum of A_q k_q(x, x'), where we are chosing to keep k_q the same for all Q
        # we then define the combined multi-output covariance as an LCM with Q = 4 (num_outputs) and A = 1/2root(4)*N(0,1) + kI for all Q
        kernel = LCM(
            X.shape[1] - 1,
            self.num_outputs,
            [
                RBF(X.shape[1] - 1),
                RBF(X.shape[1] - 1),
                RBF(X.shape[1] - 1),
                RBF(X.shape[1] - 1),
            ],
            W_rank=self.num_outputs,
        )
        likelihood = build_likelihood(Y_list, self.output_index, None)
        super().__init__(
            X,
            Y,
            kernel,
            likelihood,
            Y_metadata={"output_index": self.output_index},
            name="Genetic Drift Model",
        )
