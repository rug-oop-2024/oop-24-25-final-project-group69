from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.svm import SVC


class SupportVectorMachine(Model):
    """
    A Support Vector Machine (SVM) model class that includes model metadata 
    management and persistence through an artifact.

    Attributes:
        _model (SVC): An instance of the SVM model, configured with 
            initialization arguments.
        _artifact (Artifact): An artifact representing the SVM model, 
            storing metadata like model type, name, tags, version, and 
            asset path. This artifact is used for model tracking and 
            persistence of model metadata.
    """
    _model = None
    _artifact: Artifact

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes a Support Vector Machine (SVM) model instance and creates
        an associated artifact to store metadata and manage storage information
        for the model.

        Args:
            *args: Positional arguments passed to the SVC initializer.
            **kwargs: Keyword arguments passed to the SVC initializer.
        
        Attributes:
            _model (SVC): The Support Vector Machine model instance,
            initialized
                with the provided arguments.
            _artifact (Artifact): An artifact representing the SVM model, with
                details including type, name, asset path, tags, metadata,
                version, and data. Used for metadata management and model
                persistence.
        """
        super().__init__()
        self._type = "Classification model"
        self._model = SVC(*args, **kwargs)
        self._parameters = {
            "hyperparameters": self._model.get_params()
        }
        self._artifact = Artifact(
            type="model: Support Vector Machine",
            name="Support Vector Machine",
            asset_path=
            "autoop.core.ml.model.classification.support_vector_machine.py",
            tags=["classification", "svm"],
            metadata={},
            version="1.0.0",
            data=str(self._parameters).encode()
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit method: fits the model
        to the provided observations and ground truth

        Arguments:
            X: a 2D array with each row containing features
            for each observation
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
                "class_weight": self._model.class_weight_,
                "classes": self._model.classes_,
                "coef": self._model.coef_,
                "dual_coef": self._model.dual_coef_,
                "fit_status": self._model.fit_status_,
                "intercept": self._model.intercept_,
                "n_features_in": self._model.n_features_in_,
                "feature_names_in":
                    self._model.feature_names_in_,
                "n_iter": self._model.n_iter_,
                "support": self._model.support_,
                "support_vectors": self._model.support_vectors_,
                "n_support": self._model.n_support_,
                "probA": self._model.probA_,
                "probB": self._model.probB_,
                "shape_fit": self._model.shape_fit_
            }
        })

        self._artifact.data = str(self._parameters).encode()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict method: predicts the class labels for each observation

        Arguments:
            X: a 2D array with each
            row containing features for new observations

        Returns:
            a numpy array of predicted class labels
        """
        return self._model.predict(X)
