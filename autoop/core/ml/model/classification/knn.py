from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class KNearestNeighbors(Model):
    """
    KNN class: inherits from the Model class

    Constructor Arguments:
        Inherits those of the model class: _parameters
        model: initialized with the Sklearn KNeighborsClassifier model
        with its default arguments

    Methods:
        fit
        predict
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._model = KNeighborsClassifier(*args, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit method: fits the model on the provided labeled data

        Arguments:
            observations: a 2D array with each row containing features
            for each observation, with one column containing each feature
            ground_truth: a 1D array containing, for each observation,
            the label of the feature
            that will be predicted for new observations

        Returns:
            None
        """
        # Use the sklearn KNeighborsClassifier's fit method
        self._model.fit(X, y)
        
        # Add the attributes of the Sklearn KNeighborsClassifier model
        # and the hyperparameters using KNeighborsClassifier's get_params() method
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

    def predict(self, X: np.ndarray) -> np.ndarray: 
        """
        Predict method: predicts the label of the feature for each observation

        Arguments:
            observations: a 2D array with each row containing features
            for each new observation, with one column containing each feature

        Returns:
            a numpy array of predictions
        """
        # Use Sklearn KNeighborsClassifier's predict method
        predictions = self._model.predict(X)
        return predictions
