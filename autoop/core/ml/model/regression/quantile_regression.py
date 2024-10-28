from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.linear_model import QuantileRegressor as QuantReg


class QuantileRegressor(Model):
    """
    QuantileRegressor class: inherits from the Model class

    Constructor Arguments:
        Inherits those of the model class: _parameters
        model: initialized with the Sklearn QuantileRegressor model
        with its default arguments

    Methods:
        fit
        predict
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._model = QuantReg(*args, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit method: fits the observations by calculating the optimal parameter

        Arguments:
            observations: a 2D array with each row containing features
            for each observation, with one column containing each feature
            ground_truth: a 1D array containing, for each observation,
            the value of the feature
            that will be predicted for new observations

        Returns:
            None
        """
        # Use the sklearn QuantileRegressor's fit method
        self._model.fit(X, y)
        
        # Add the attributes of the Sklearn QuantileRegressor model
        # and the hyperparameters using QuantileRegressor's get_params() method
        self._parameters = {
            "strict parameters": {
                "coef": self._model.coef_,
                "intercept": self._model.intercept_,
                "n_features_in": self._model.n_features_in_,
                "features_names_in": self._model.feature_names_in_,
                "n_iter": self._model.n_iter_
            },
            "hyperparameters": self._model.get_params()
        }

    def predict(self, X: np.ndarray) -> np.ndarray: 
        """
        Predict method: predicts the value of the feature for each observation

        Arguments:
            observations: a 2D array with each row containing features
            for each new observation, with one column containing each feature

        Returns:
            a numpy array of predictions
        """
        # Use Sklearn QuantileRegressor's predict method
        predictions = self._model.predict(X)
        return predictions
