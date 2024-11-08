import pickle
from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.linear_model import Lasso as SklearnLasso
from autoop.core.ml.artifact import Artifact
from sklearn.preprocessing import StandardScaler

class MultipleLinearRegression(Model):
    """
    LassoRegression class: inherits from the Model class

    Constructor Arguments:
        Inherits those of the model class: _parameters
        _model: initialized with an instance of the Sklearn Lasso model
        with its default arguments

    Methods:
        fit
        predict
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes a Lasso Regressor model instance using SklearnLasso 
        and sets up an artifact to store model metadata and manage storage info.

        Args:
            *args: Positional arguments passed to the SklearnLasso initializer.
            **kwargs: Keyword arguments passed to the SklearnLasso initializer.

        Attributes:
            _type (str): The type of model, in this case, "regression".
            _model (SklearnLasso): The Lasso Regressor model instance,
            initialized with the provided arguments for Lasso.
            _parameters (dict): A dictionary holding the hyperparameters of 
            the model, initialized with the Lasso model's parameters.
            _target_scaler (StandardScaler or None): A scaler for the target variable,
            initialized as None but used for inverse transformation during prediction.
        """
        super().__init__()
        self._type = "regression"
        self._model = SklearnLasso(alpha=0.001, *args, **kwargs)
        self._parameters = {
            "hyperparameters": self._model.get_params()
        }
        self._target_scaler = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit method: fits the observations by calculating the optimal parameter.

        Arguments:
            X: A 2D array where each row contains features for each observation.
            y: A 1D array containing the ground truth values for the observations.

        Returns:
            None
        """
        X = np.asarray(X)
        
        # Use the sklearn Lasso's fit method
        self._model.fit(X, y)

        self._target_scaler = StandardScaler()
        self._target_scaler.fit(y.reshape(-1, 1))
        
        # Add the coef_ and intercept_ parameters
        # of the Sklearn Lasso model
        # and the hyperparameters using Lasso's
        # get_params() method
        self._parameters.update({
            "strict parameters": {
                "coef": self._model.coef_,
                "intercept": self._model.intercept_
            }
        })

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict method: predicts the value of the feature for each observation.

        Arguments:
            X: A 2D array with each row containing features 
            for new observations.

        Returns:
            A numpy array of predicted values for each observation.
        """
        # Use Sklearn Lasso's predict method
        predictions = self._model.predict(X)

        # Inverse transform the predictions using the scaler
        predictions = self._target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        return predictions
