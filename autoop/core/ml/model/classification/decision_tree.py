from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DecTreeClass


class DecisionTreeClassifier(Model):
    """
    DecisionTreeClassifier class: inherits from the Model class

    Constructor Arguments:
        Inherits those of the model class: _parameters
        model: initialized with the Sklearn DecisionTreeClassifier
        model with its default arguments

    Methods:
        fit
        predict
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes a DecisionTreeClassifier model instance.

        This constructor initializes a Sklearn DecisionTreeClassifier model
        and sets up the model's hyperparameters in the _parameters attribute.
        It is called with any additional arguments passed to the parent class
        initializer, allowing customization of the DecisionTreeClassifier's
        configuration.

        Args:
            *args: Positional arguments passed to the DecisionTreeClassifier
            initializer.
            **kwargs: Keyword arguments passed to the DecisionTreeClassifier
            initializer.

        Attributes:
            _type (str): The type of model, in this case, "classification".
            _model (DecTreeClass): The Sklearn DecisionTreeClassifier model
            instance, configured with the provided initialization arguments.
            _parameters (dict): A dictionary holding the hyperparameters
            of the model, initialized with the DecisionTreeClassifier's
            parameters.
        """
        super().__init__()
        self._type = "classification"
        self._model = DecTreeClass(*args, **kwargs)
        self._parameters = {
            "hyperparameters": self._model.get_params()
        }

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
                "n_outputs": self._model.n_outputs_,
                "tree": self._model.tree_
            }
        })

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
