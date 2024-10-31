from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class KNearestNeighbors(Model):
    """
    KNearestNeighbors class: inherits from the Model class

    Constructor Arguments:
        Inherits those of the model class: _parameters
        model: initialized with the Sklearn KNeighborsClassifier model
        with its default arguments

    Methods:
        fit
        predict
    """
    _model = None
    _artifact: Artifact

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._model = KNeighborsClassifier(*args, **kwargs)
        self._artifact = Artifact(
            type="model: K Nearest Neighbors",
            name="K Nearest Neighbors",
            asset_path="autoop.core.ml.model.classification.k_nearest_neighbors.py",
            tags=["classification", "knn"],
            metadata={},
            version="1.0.0",
            data=None
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit method: fits the model on the provided labeled data

        Arguments:
            X: a 2D array with each row containing features for each observation
            y: a 1D array containing the labels for the observations

        Returns:
            None
        """
        self._model.fit(X, y)

        # Add model parameters to _parameters
        self._parameters = {
            "strict parameters": {
                "classes": self._model.classes_,
                "effective_metric": self._model.effective_metric_,
                "effective_metric_params": self._model.effective_metric_params_,
                "n_features_in": self._model.n_features_in_,
                "feature_names_in": self._model.feature_names_in_,
                "n_samples_fit": self._model.n_samples_fit_,
                "outputs_2d": self._model.outputs_2d_
            },
            "hyperparameters": self._model.get_params()
        }

        self._artifact.data = str(self._parameters).encode()

    def predict(self, X: np.ndarray) -> np.ndarray: 
        """
        Predict method: predicts the label for each observation

        Arguments:
            X: a 2D array with each row containing features for new observations

        Returns:
            a numpy array of predicted labels
        """
        return self._model.predict(X)
