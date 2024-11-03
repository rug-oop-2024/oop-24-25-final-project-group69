from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DecTreeClass


class DecisionTreeClassifier(Model):
    """
    DecisionTreeClassifier class: inherits from the Model class

    Constructor Arguments:
        Inherits those of the model class: _parameters
        model: initialized with the Sklearn DecisionTreeClassifier
        model
        with its default arguments

    Methods:
        fit
        predict
    """
    _model = None
    _artifact: Artifact

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._type = "Classification model"
        self._model = DecTreeClass(*args, **kwargs)
        self._parameters = {
            "hyperparameters": self._model.get_params()
        }
        self._artifact = Artifact(
            type="model: Decision Tree Classifier",
            name="Decision Tree Classifier",
            asset_path=
            "autoop.core.ml.model.classification.decision_tree_classifier.py",
            tags=["classification", "decision tree"],
            metadata={},
            version="1.0.0",
            data=str(self._parameters).encode()
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit method: fits the model
        to the provided observations and ground truth

        Arguments:
            X: a 2D array with each row containing
            features for each observation
            y: a 1D array containing the class labels
            for each observation

        Returns:
            None
        """
        X = np.asarray(X)

        self._model.fit(X, y)

        # Add model parameters to _parameters
        self._parameters.update({
            "strict parameters": {
                "classes": self._model.classes_,
                "feature_importances": self._model.feature_importances_,
                "max_features": self._model.max_features_,
                "n_classes": self._model.n_classes_,
                "n_features_in": self._model.n_features_in_,
                "feature_names_in": self._model.feature_names_in_,
                "n_outputs": self._model.n_outputs_,
                "tree": self._model.tree_
            }
        })

        self._artifact.data = str(self._parameters).encode()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict method: predicts the class labels for each observation

        Arguments:
            X: a 2D array with each row containing
            features for new observations

        Returns:
            a numpy array of predicted class labels
        """
        return self._model.predict(X)
