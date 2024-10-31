
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.multiple_linear_regression import MultipleLinearRegression
from autoop.core.ml.model.regression.quantile_regression import QuantileRegressor
from autoop.core.ml.model.regression.logistic_regression import LogisticRegression
from autoop.core.ml.model.classification.decision_tree import DecisionTreeClassifier
from autoop.core.ml.model.classification.knn import KNearestNeighbors
from autoop.core.ml.model.classification.svm import SupportVectorMachine



REGRESSION_MODELS = [
    "multiple_linear_regression",
    "quantile_regressor",
    "logistic_regression"
]

CLASSIFICATION_MODELS = [
    "knn",
    "svm",
    "decision_tree"
]

def get_model(model_name: str) -> Model:
    """Factory function to get a model by name.
    Args:
        model_name(str): the name of the model as a string
    Returns:
        Model: a model instance given its str name
    """
    if model_name == "multiple_linear_regression":
        return MultipleLinearRegression()
    if model_name == "quantile_regression":
        return QuantileRegressor()
    if model_name == "logistic_regression":
        return LogisticRegression()
    if model_name == "knn":
        return KNearestNeighbors()
    if model_name == "svm":
        return SupportVectorMachine()
    if model_name == "decision_tree":
        return DecisionTreeClassifier()
