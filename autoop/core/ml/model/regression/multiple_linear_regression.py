from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.linear_model import Lasso as SklearnLasso


class MultipleLinearRegression(Model):
    """
    MultipleLinearRegression class: inherits from the Model class

    Constructor Arguments:
        Inherits those of the model class: _parameters
        _model: initialized with an instance of the
        Sklearn Lasso model with its default arguments

    Methods:
        fit
        predict
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._model = SklearnLasso(*args, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit method: fits the observations by calculating the optimal parameter

        Arguments:
            observations: a 2D array with each row containing features
            for each observation, with one column containing each feature
            ground_truth: a 1D array containing, for each observation,
            the value of the feature
            that will be predicted for new observations

        Returns:
            None
        """
        # Use the sklearn Lasso's fit method
        self._model.fit(X, y)
        
        # Add the coef_ and intercept_ parameters of the Sklearn Lasso model
        # and the hyperparameters using Lasso's get_params() method
        self._parameters = {
            "strict parameters": {
                "coef": self._model.coef_,
                "intercept": self._model.intercept_
            },
            "hyperparameters": self._model.get_params()
        }

    def predict(self, X: np.ndarray) -> np.ndarray: 
        """
        Predict method: predicts the value of the feature for each observation

        Arguments:
            observations: a 2D array with each row containing features
            for each new observation, with one column containing each feature

        Returns:
            a numpy array of predictions
        """
        # Use Sklearn Lasso's predict method
        predictions = self._model.predict(X)
        return predictions