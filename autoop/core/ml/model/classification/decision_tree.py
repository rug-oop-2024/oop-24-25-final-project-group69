from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DecTreeClass


class DecisionTreeClassifier(Model):
    """
    DecisionTree class: inherits from the Model class

    Constructor Arguments:
        Inherits those of the model class: _parameters
        model: initialized with the Sklearn DecisionTreeClassifier model
        with its default arguments

    Methods:
        fit
        predict

    Properties:
        parameters
    """
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._model = DecTreeClass(*args, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit method: fits the model to the provided observations and ground truth

        Arguments:
            X: a 2D array with each row containing features for each observation
            y: a 1D array containing the class labels for each observation

        Returns:
            None
        """
        # Use Sklearn DecTreeClassifier's fit method
        self._model.fit(X, y)

        # Add the attributes of the Sklearn DecTreeClassifier model
        # and the hyperparameters using DecTreeClassifier's get_params() method
        self._parameters = {
            "strict parameters": {
                "classes": self._model.classes_,
                "feature_importances": self._model.feature_importances_,
                "max_features": self._model.max_features_,
                "n_classes": self._model.n_classes_,
                "n_features_in": self._model.n_features_in_,
                "feature_names_in": self._model.feature_names_in_,
                "n_outputs": self._model.n_outputs_,
                "tree": self._model.tree_,
            },
            "hyperparameters": self._model.get_params()
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict method: predicts the class labels for each observation

        Arguments:
            X: a 2D array with each row containing features for new observations

        Returns:
            A list of predicted class labels
        """
        # Use Sklearn DecTreeClassifier's predict method
        return self._model.predict(X)
