from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.svm import SVC


class SupportVectorMachine(Model):
    """
    SupportVectorMachine class: inherits from the Model class

    Constructor Arguments:
        Inherits those of the model class: _parameters
        model: initialized with the Sklearn SVC model
        with its default arguments

    Methods:
        fit
        predict

    Properties:
        parameters
    """
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._model = SVC(*args, **kwargs)
        

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit method: fits the model to the provided observations and ground truth

        Arguments:
            X: a 2D array with each row containing features for each observation
            y: a 1D array containing the class labels for each observation

        Returns:
            None
        """
        # Use the sklearn SVM's fit method
        self._model.fit(X, y)

        # Add the attributes of the Sklearn SVM model
        # and the hyperparameters using SVM's get_params() method
        self._parameters = {
            "strict parameters": {
                "class_weight": self._model.class_weight_,
                "classes": self._model.classes_,
                "coef": self._model.coef_,
                "dual_coef": self._model.dual_coef_,
                "fit_status": self._model.fit_status_,
                "intercept": self._model.intercept_,
                "n_features_in": self._model.n_features_in_,
                "feature_names_in": self._model.feature_names_in_,
                "n_iter": self._model.n_iter_,
                "support": self._model.support_,
                "support_vectors": self._model.support_vectors_,
                "n_support": self._model.n_support_,
                "probA": self._model.probA_,
                "probB": self._model.probB_,
                "shape_fit": self._model.probB_
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
        # Use Sklearn SVM's predict method
        return self._model.predict(X)
