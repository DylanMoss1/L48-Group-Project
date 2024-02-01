import numpy as np
from emukit.core.loop import ModelUpdater, LoopState, UserFunctionResult

from ..models import CoupledGPModel


class CustomUserFunctionResult(UserFunctionResult):
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        input_data=None,
        output_data=None,
        **kwargs
    ) -> None:
        """
        :param X: Function input. Shape: (function input dimension,)
        :param Y: Function output(s). Shape: (function output dimension,)
        :param kwargs: Extra outputs of the UserFunction to store. Shape: (extra output dimension,)
        """
        if X.ndim != 1:
            raise ValueError(
                "x is expected to be 1-dimensional, actual dimensionality is {}".format(
                    X.ndim
                )
            )

        if Y.ndim != 1:
            raise ValueError(
                "y is expected to be 1-dimensional, actual dimensionality is {}".format(
                    Y.ndim
                )
            )

        self.extra_outputs = dict()
        for key, val in kwargs.items():
            if val.ndim != 1:
                raise ValueError(
                    "Key word arguments must be 1-dimensional but {} is {}d".format(
                        key, val.ndim
                    )
                )
            self.extra_outputs[key] = val

        self.X = X
        self.Y = Y
        self.input_data = input_data
        self.output_data = output_data


class CustomIntervalUpdater(ModelUpdater):
    """Updates hyper-parameters every nth iteration, where n is defined by the user"""

    def __init__(self, model: CoupledGPModel, interval: int = 1) -> None:
        """
        :param model: Emukit emulator model
        :param interval: Number of function evaluations between optimizing model hyper-parameters
        :param targets_extractor_fcn: A function that takes in loop state and returns the training targets.
                                      Defaults to a function returning loop_state.Y
        """
        self.model = model
        self.interval = interval

    def update(self, loop_state: LoopState) -> None:
        """
        :param loop_state: Object that contains current state of the loop
        """
        data_input = np.vstack(
            [result.input_data for result in loop_state.results]
        )
        data_output = np.vstack(
            [result.output_data for result in loop_state.results]
        )
        self.model.set_data(data_input, data_output)
        self.model.set_results(loop_state.X, loop_state.Y)
        if (loop_state.iteration % self.interval) == 0:
            self.model.optimize()
