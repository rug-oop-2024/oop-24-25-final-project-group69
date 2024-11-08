from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RandForClassifier


class RandomForestClassifier(Model):
    """
    RandomForestClassification class: inherits from the Model class

    Constructor Arguments:
        Inherits those of the model class: _parameters
        model: initialized with the Sklearn RandomForestClassifier model
        with its default arguments

    Methods:
        fit
        predict
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes a Random Forest Classifier model instance and
        prepares an associated artifact for tracking model metadata
        and storage information.

        Args:
            *args: Positional arguments passed
            to the RandomForestClassifier initializer.
            **kwargs: Keyword arguments passed
            to the RandomForestClassifier initializer.

        Attributes:
            _model (RandomForestClassifier):
            The Random Forest Classifier model instance
            initialized with provided arguments.
        """
        super().__init__()
        self._type = "classification"
        self._model = RandForClassifier(*args, **kwargs)
        self._parameters = {
            "hyperparameters": self._model.get_params()
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit method: fits the model on the provided labeled data

        Arguments:
            X: a 2D array with each row containing
            features for each observation
            y: a 1D array containing the labels for the observations

        Returns:
            None
        """
        X = np.asarray(X)

        self._model.fit(X, y)

        # Add model parameters to _parameters
        self._parameters.update({
            "strict parameters": {
                "estimator_": str(self._model.estimator_),
                "estimators_": len(self._model.estimators_),
                "classes_": self._model.classes_,
                "n_classes_": self._model.n_classes_,
                "n_features_in_": self._model.n_features_in_,
                "n_outputs_": self._model.n_outputs_,
                "feature_importances_": self._model.feature_importances_
            }
        })

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict method: predicts the label for each observation.

        Arguments:
            X: a 2D array with each row containing features\
            for new observations.

        Returns:
            a numpy array of predicted labels.
        """
        return self._model.predict(X)
