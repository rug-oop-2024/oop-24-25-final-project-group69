import pickle
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

    def _init_(self, *args, **kwargs) -> None:
        """
        Initializes a K Nearest Neighbors (KNN) model instance.

        This constructor initializes a Sklearn KNeighborsClassifier model
        and sets up the model's hyperparameters in the _parameters attribute.
        It is called with any additional arguments passed to the parent class 
        initializer, allowing customization of the KNeighborsClassifier's 
        configuration.

        Args:
            *args: Positional arguments passed to the KNeighborsClassifier 
            initializer.
            **kwargs: Keyword arguments passed to the KNeighborsClassifier 
            initializer.

        Attributes:
            _type (str): The type of model, in this case, "classification".
            _model (KNeighborsClassifier): The Sklearn KNeighborsClassifier 
            model instance, configured with the provided initialization 
            arguments.
            _parameters (dict): A dictionary holding the hyperparameters 
            of the model, initialized with the KNeighborsClassifier's 
            parameters.
        """
        super()._init_()
        self._type = "classification"
        self._model = KNeighborsClassifier(*args, **kwargs)
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
                "classes": self.model.classes,
                "effective_metric": self.model.effective_metric,
                "effective_metric_params":
                    self.model.effective_metric_params,
                "n_features_in": self.model.n_features_in,
                "n_samples_fit": self.model.n_samples_fit,
                "outputs_2d": self.model.outputs_2d
            }
        })


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