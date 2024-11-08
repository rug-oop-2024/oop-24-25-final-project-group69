import pickle
from autoop.core.ml.model.model import Model
from autoop.core.ml.artifact import Artifact
from sklearn.linear_model import ARDRegression as SklearnARDRegression
import numpy as np

class ARDRegression(Model):
    """
    ARDRegressionWrapper class: inherits from the Model class.

    Constructor Arguments:
        Inherits those of the Model class: _parameters.
        _model: initialized with an instance of the Sklearn ARDRegression model
        with its default arguments.

    Methods:
        fit
        predict
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes an ARD Regressor model instance using SklearnARDRegression 
        and sets up an artifact to store model metadata and manage storage info.

        Args:
            *args: Positional arguments passed to the SklearnARDRegression 
            initializer.
            **kwargs: Keyword arguments passed to the SklearnARDRegression 
            initializer.

        Attributes:
            _type (str): The type of model, in this case, "regression".
            _model (SklearnARDRegression): The ARD Regressor model instance, 
            initialized with the provided arguments for the ARDRegression.
            _parameters (dict): A dictionary holding the hyperparameters of 
            the model, initialized with the ARDRegression model's parameters.
            _target_scaler (None): A placeholder for a potential target scaler 
            to be used in preprocessing.
        """
        super().__init__()
        self._type = "regression"
        self._model = SklearnARDRegression(*args, **kwargs)
        self._parameters = {
            "hyperparameters": self._model.get_params()
        }
        self._target_scaler = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit method: fits the observations by calculating the optimal parameter.

        Arguments:
            X: a 2D array with each row containing features for each observation.
            y: a 1D array containing the ground truth values for the observations.

        Returns:
            None
        """
        X = np.asarray(X)
        
        # Use the sklearn ARDRegression's fit method
        self._model.fit(X, y)

        # Add model parameters to _parameters
        self._parameters.update({
            "strict parameters": {
                "coef": self._model.coef_,
                "intercept": self._model.intercept_,
                "alpha": self._model.alpha_,
                "lambda": self._model.lambda_
            }
        })

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict method: predicts the value of the feature for each observation.

        Arguments:
            X: a 2D array with each row containing features for new observations.

        Returns:
            A numpy array of predictions.
        """
        return self._model.predict(X)
