import pickle
from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.linear_model import Ridge as SklearnRidge

from autoop.core.ml.artifact import Artifact


class RidgeRegression(Model):
    """
    RidgeRegression class: inherits from the Model class

    Constructor Arguments:
        Inherits those of the model class: _parameters
        _model: initialized with an instance of the
        Sklearn Ridge model with its default arguments

    Methods:
        fit
        predict
    """

    _model = None
    _artifact: Artifact

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes a Ridge Regressor model
        instance using SklearnRidge
        and sets up an artifact to store model metadata
        and manage storage info.

        Args:
            *args: Positional arguments passed to
            the SklearnRidge initializer.
            **kwargs: Keyword arguments passed to the SklearnRidge initializer.

        Attributes:
            _model (SklearnRidge): The Ridge Regressor model instance,
                set up with provided initialization arguments.
            _artifact (Artifact): An artifact representing the regressor model,
                storing metadata such as type, name, tags, version, asset path,
                and experiment identifiers for model management and persistence.
        """
        super().__init__()
        self._type = "regression"
        self._model = SklearnRidge(*args, **kwargs)
        self._parameters = {
            "hyperparameters": self._model.get_params()
        }
        self._artifact = Artifact(
            type="model: Ridge Regressor",
            name="Ridge Regressor",
            asset_path=
            "autoop.core.ml.model.regression.ridge_regression.py",
            tags=["regression", "ridge"],
            metadata={
                "experiment_id": "exp-123fbdiashdb",
                "run_id": "run-12378yufdh89afd",
            },
            version="1.0.0",
            data=pickle.dumps(self._parameters)
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit method: fits the observations by calculating
        the optimal parameter

        Arguments:
            observations: a 2D array with each row containing features
            for each observation, with one column containing each feature
            ground_truth: a 1D array containing, for each observation,
            the value of the feature
            that will be predicted for new observations

        Returns:
            None
        """
        X = np.asarray(X)
        
        # Use the sklearn Ridge's fit method
        self._model.fit(X, y)
        
        # Add the coef_ and intercept_ parameters
        # of the Sklearn Ridge model
        # and the hyperparameters using Ridge's
        # get_params() method
        self._parameters.update({
            "strict parameters": {
                "coef": self._model.coef_,
                "intercept": self._model.intercept_,
            }
        })

        self._artifact.data = pickle.dumps(self._parameters)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict method: predicts the value of the
        feature for each observation

        Arguments:
            observations: a 2D array with each
            row containing features
            for each new observation, with one
            column containing each feature

        Returns:
            a numpy array of predictions
        """
        # Use Sklearn Ridge's predict method
        predictions = self._model.predict(X)
        return predictions
