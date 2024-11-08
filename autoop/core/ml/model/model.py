from abc import abstractmethod
import pickle
from pydantic import PrivateAttr
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal


class Model():
    """
    Model class: inherits from the ABC class

    Constructor Arguments:
        None

    Methods:
        fit
        predict
    """
    _model = None
    _parameters: dict
    _type: str
    _artifact: Artifact

    @property
    def type(self) -> str:
        """Getter for _type

        Returns:
            str: deepcopy of _type
        """
        return self._type

    @property
    def parameters(self) -> dict:
        """Getter for _parameters

        Returns:
            str: deepcopy of _parameters
        """
        return deepcopy(self._parameters)

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Abstract method fit: fits the observations for a given model

        Arguments:
            observations: a 2D array with each row containing features
            for each observation, with one column containing each feature
            ground_truth: a 1D array containing, for each observation,
            the value of the feature that
            will be predicted for new observations

        Returns:
                None
            """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Abstract method predict: predicts
        the value of the feature for each observation

        Arguments:
            observations: a 2D array with each row containing
            features for each new observation,
            with one column containing each feature

        Returns:
            a list of predictions
        """
        pass

    def to_artifact(self, name: str) -> Artifact:
        data = pickle.dumps(self._parameters)
        self._artifact = Artifact(name=name, data=data)
        return self._artifact
