from typing import List

import numpy as np
from GPy.util.multioutput import build_XY, build_likelihood, LCM
from GPy.kern import Linear, RBF
from GPy.core.gp import GP


def get_next_trait_kernel(X, trait_index_in_input): 
    trait_kernel = Linear(X.shape[1] - 1, active_dims=[trait_index_in_input])
    remaining_inputs_kernel = RBF(X.shape[1] - 1, lengthscale=0.1, variance=1, active_dims=[i for i in range(len(X)) if i != trait_index_in_input])
    next_trait_kernel = trait_kernel + remaining_inputs_kernel
    return next_trait_kernel

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
        
        # kernel_rbf_1 = GPy.kern.RBF(input_dim=1, lengthscale=0.1, variance=1, active_dims=[0])
        # remaining_inputs_kernel = RBF(X.shape[1], lengthscale=0.1, variance=1, active_dims=[i for i in range(len(X)) if i != 1])

        next_size_kernel = get_next_trait_kernel(X, 2)
        next_speed_kernel = get_next_trait_kernel(X, 3)
        next_vision_kernel = get_next_trait_kernel(X, 4)
        next_aggression_kernel = get_next_trait_kernel(X, 5)

        # -1 is because X has an addition column at the end to specify the output index to return, which doesn't matter for the kernel (not an official input)
        kernel = LCM(
            X.shape[1] - 1,
            self.num_outputs,
            [
                next_size_kernel,
                next_speed_kernel,
                next_vision_kernel,
                next_aggression_kernel,
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


