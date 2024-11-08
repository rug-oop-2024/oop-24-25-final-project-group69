from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.linear_model import Ridge as SklearnRidge


class RidgeRegression(Model):
    """
    RidgeRegression class: inherits from the Model class

    Constructor Arguments:
        Inherits those of the model class: _parameters
        _model: initialized with an instance of the Sklearn Ridge model
        with its default arguments

    Methods:
        fit
        predict
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes a Ridge Regressor model instance using SklearnRidge
        and sets up an artifact to store model metadata and manage storage info.

        Args:
            *args: Positional arguments passed to the SklearnRidge
            initializer.
            **kwargs: Keyword arguments passed to the SklearnRidge
            initializer.

        Attributes:
            _type (str): The type of model, in this case, "regression".
            _model (SklearnRidge): The Ridge Regressor model instance,
            initialized with the provided arguments for Ridge.
            _parameters (dict): A dictionary holding the
            hyperparameters of
            the model, initialized with the Ridge model's parameters.
        """
        super().__init__()
        self._type = "regression"
        self._model = SklearnRidge(*args, **kwargs)
        self._parameters = {
            "hyperparameters": self._model.get_params()
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit method: fits the observations by
        calculating the optimal parameter.

        Arguments:
            X: A 2D array where each row contains
            features for each observation.
            y: A 1D array containing the ground
            truth values for the observations.

        Returns:
            None
        """
        X = np.asarray(X)

        # Use the sklearn Ridge's fit method
        self._model.fit(X, y)

        # Add the coef_ and intercept_ parameters
        # of the Sklearn Ridge model
        # and the hyperparameters using Ridge's
        # get_params() method
        self._parameters.update({
            "strict parameters": {
                "coef": self._model.coef_,
                "intercept": self._model.intercept_,
            }
        })

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict method: predicts the value of the feature for each observation.

        Arguments:
            X: A 2D array with each row containing features
            for new observations.

        Returns:
            A numpy array of predicted values for each observation.
        """
        # Use Sklearn Ridge's predict method
        return self._model.predict(X)
