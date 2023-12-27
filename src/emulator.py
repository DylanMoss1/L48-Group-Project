from typing import Tuple

import numpy as np

from emukit.core.interfaces import IModel, IModelWithNoise

class GeneticDriftModel(IModel, IModelWithNoise):
    def __init__(self):
        pass
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance values for given points

        :param X: array of shape (n_points x n_inputs) of points to run prediction for
        :return: Tuple of mean and variance which are 2d arrays of shape (n_points x n_outputs)
        """
        raise NotImplementedError
    
    def predict_noiseless(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        For given points X, predict mean and variance of the output without observation noise.

        :param X: array of shape (n_points x n_inputs) of points to run prediction for
        :return: Tuple of mean and variance which are 2d arrays of shape (n_points x n_outputs)
        """
        raise NotImplementedError

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
        raise NotImplementedError

    @property
    def X(self):
        raise NotImplementedError

    @property
    def Y(self):
        raise NotImplementedError

class PopulationModel(IModel, IModelWithNoise):
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance values for given points

        :param X: array of shape (n_points x n_inputs) of points to run prediction for
        :return: Tuple of mean and variance which are 2d arrays of shape (n_points x n_outputs)
        """
        raise NotImplementedError
    
    def predict_noiseless(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        For given points X, predict mean and variance of the output without observation noise.

        :param X: array of shape (n_points x n_inputs) of points to run prediction for
        :return: Tuple of mean and variance which are 2d arrays of shape (n_points x n_outputs)
        """
        raise NotImplementedError

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
        raise NotImplementedError

    @property
    def X(self):
        raise NotImplementedError

    @property
    def Y(self):
        raise NotImplementedError