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
        """
        Initializes a K Nearest Neighbors (KNN)
        model instance and prepares an associated artifact
        for tracking model metadata and storage information.

        Args:
            *args: Positional arguments passed
            to the KNeighborsClassifier initializer.
            **kwargs: Keyword arguments passed
            to the KNeighborsClassifier initializer.
        
        Attributes:
            _model (KNeighborsClassifier): The K Nearest Neighbors
            model instance initialized with provided arguments.
            _artifact (Artifact): An artifact representing this KNN model,
            including type, name, asset path, tags,
                                metadata, version, and data.
                                Used for model metadata management
                                and persistence.
        """
        super().__init__()
        self._type = "Classification model"
        self._model = KNeighborsClassifier(*args, **kwargs)
        self._parameters = {
            "hyperparameters": self._model.get_params()
        }
        self._artifact = Artifact(
            type="model: K Nearest Neighbors",
            name="K Nearest Neighbors",
            asset_path=
            "autoop.core.ml.model.classification.k_nearest_neighbors.py",
            tags=["classification", "knn"],
            metadata={},
            version="1.0.0",
            data=str(self._parameters).encode()
        )

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
                "classes": self._model.classes_,
                "effective_metric": self._model.effective_metric_,
                "effective_metric_params":
                    self._model.effective_metric_params_,
                "n_features_in": self._model.n_features_in_,
                "feature_names_in": self._model.feature_names_in_,
                "n_samples_fit": self._model.n_samples_fit_,
                "outputs_2d": self._model.outputs_2d_
            }
        })

        self._artifact.data = str(self._parameters).encode()

    def predict(self, X: np.ndarray) -> np.ndarray: 
        """
        Predict method: predicts the label for each observation

        Arguments:
            X: a 2D array with each row containing features
            for new observations

        Returns:
            a numpy array of predicted labels
        """
        return self._model.predict(X)
