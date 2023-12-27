from typing import Tuple

import numpy as np

from GPy.models import GPRegression
from emukit.core.interfaces import IModel, IModelWithNoise

class GeneticDriftModel(IModel, IModelWithNoise):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        """Initializes a GP model using GPy with an RBF kernel (variance 1, lengthscale 1) and Gaussian likelihood (noise variance 1)

        Args:
            X (np.ndarray): array of shape (n_points x n_inputs) of initial points
            Y (np.ndarray): array of shape (n_points x n_outputs) of initial points
        """
        self.model = GPRegression(X, Y)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance values for given points

        :param X: array of shape (n_points x n_inputs) of points to run prediction for
        :return: Tuple of mean and variance which are 2d arrays of shape (n_points x n_outputs)
        """
        return self.model.predict(X)
    
    def predict_noiseless(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        For given points X, predict mean and variance of the output without observation noise.

        :param X: array of shape (n_points x n_inputs) of points to run prediction for
        :return: Tuple of mean and variance which are 2d arrays of shape (n_points x n_outputs)
        """
        return self.model.predict(X, include_likelihood=False)

    def optimize(self) -> None:
        """
        Optimize hyper-parameters of model
        """
        raise NotImplementedError

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Sets training data in model

        :param X: new points
        :param Y: function values at new points X
        """
        self.model.set_XY(X, Y)

    @property
    def X(self):
        return self.model.X

    @property
    def Y(self):
        return self.model.Y

class PopulationModel(IModel, IModelWithNoise):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        """Initializes a GP model using GPy with an RBF kernel (variance 1, lengthscale 1) and Gaussian likelihood (noise variance 1)

        Args:
            X (np.ndarray): array of shape (n_points x n_inputs) of initial points
            Y (np.ndarray): array of shape (n_points x n_outputs) of initial points
        """
        self.model = GPRegression(X, Y)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance values for given points

        :param X: array of shape (n_points x n_inputs) of points to run prediction for
        :return: Tuple of mean and variance which are 2d arrays of shape (n_points x n_outputs)
        """
        return self.model.predict(X)
    
    def predict_noiseless(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        For given points X, predict mean and variance of the output without observation noise.

        :param X: array of shape (n_points x n_inputs) of points to run prediction for
        :return: Tuple of mean and variance which are 2d arrays of shape (n_points x n_outputs)
        """
        return self.model.predict(X, include_likelihood=False)

    def optimize(self) -> None:
        """
        Optimize hyper-parameters of model
        """
        raise NotImplementedError

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Sets training data in model

        :param X: new points
        :param Y: function values at new points X
        """
        self.model.set_XY(X, Y)

    @property
    def X(self):
        return self.model.X

    @property
    def Y(self):
        return self.model.Y