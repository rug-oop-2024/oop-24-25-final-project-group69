from autoop.core.ml.artifact import Artifact
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
    _model = None
    _artifact: Artifact

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._model = QuantReg(*args, **kwargs)
        self._artifact = Artifact(
            type="model: Quantile Regressor",
            name="Quantile Regressor",
            asset_path="autoop.core.ml.model.regression.quantile_regressor.py",
            tags=["regression", "quantile"],
            metadata={},
            version="1.0.0",
            data=None
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit method: fits the observations by calculating the optimal parameter

        Arguments:
            X: a 2D array with each row containing features for each observation
            y: a 1D array containing the ground truth values for the observations

        Returns:
            None
        """
        self._model.fit(X, y)

        # Add model parameters to _parameters
        self._parameters = {
            "strict parameters": {
                "coef": self._model.coef_,
                "intercept": self._model.intercept_,
                "n_features_in": self._model.n_features_in_,
                "feature_names_in": self._model.feature_names_in_,
                "n_iter": self._model.n_iter_
            },
            "hyperparameters": self._model.get_params()
        }

        self._artifact.data = str(self._parameters).encode()

    def predict(self, X: np.ndarray) -> np.ndarray: 
        """
        Predict method: predicts the value of the feature for each observation

        Arguments:
            X: a 2D array with each row containing features for new observations

        Returns:
            a numpy array of predictions
        """
        return self._model.predict(X)
