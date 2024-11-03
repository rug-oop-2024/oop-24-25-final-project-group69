from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.linear_model import LogisticRegression as LogReg


class LogisticRegression(Model):
    """
    LogisticRegression class: inherits from the Model class

    Constructor Arguments:
        Inherits those of the model class: _parameters
        model: initialized with the Sklearn LogisticRegression model
        with its default arguments

    Methods:
        fit
        predict
    """
    _model = None
    _artifact: Artifact

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes a Logistic Regression model instance and creates an artifact
        to store metadata and manage storage information for the model.

        Args:
            *args: Positional arguments passed to the LogReg initializer.
            **kwargs: Keyword arguments passed to the LogReg initializer.

        Attributes:
            _model (LogReg): The Logistic Regression model instance, set up
                with provided initialization arguments.
            _artifact (Artifact): An artifact representing the Logistic
                Regression model, storing metadata such as type, name, tags,
                asset path, version, and other details for model management.
        """
        super().__init__()
        self._type = "Regression model"
        self._model = LogReg(*args, **kwargs)
        self._parameters = {
            "hyperparameters": self._model.get_params()
        }
        self._artifact = Artifact(
            type="model: Logistic Regression",
            name="Logistic Regression",
            asset_path=
            "autoop.core.ml.model.classification.logistic_regression.py",
            tags=["classification", "logistic"],
            metadata={},
            version="1.0.0",
            data=str(self._parameters).encode()
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit method: fits
        the observations by calculating the optimal parameter

        Arguments:
            X: a 2D array with each row containing
            features for each observation
            y: a 1D array containing the ground
            truth labels for the observations

        Returns:
            None
        """
        X = np.asarray(X)
        
        self._model.fit(X, y)

        # Add model parameters to _parameters
        self._parameters.update({
            "strict parameters": {
                "coef": self._model.coef_,
                "intercept": self._model.intercept_
            }
        })

        self._artifact.data = str(self._parameters).encode()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict method: predicts the label for each observation

        Arguments:
            X: a 2D array with each
            row containing features for new observations

        Returns:
            a numpy array of predicted labels
        """
        return self._model.predict(X)
