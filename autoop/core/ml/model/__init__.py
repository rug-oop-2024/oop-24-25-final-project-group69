from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.multiple_linear_regression import MultipleLinearRegression
from autoop.core.ml.model.regression.quantile_regression import QuantileRegressor
from autoop.core.ml.model.regression.logistic_regression import LogisticRegression
from autoop.core.ml.model.classification.decision_tree import DecisionTreeClassifier
from autoop.core.ml.model.classification.knn import KNearestNeighbors
from autoop.core.ml.model.classification.svm import SupportVectorMachine


REGRESSION_MODELS = [
    "Multiple Linear Regression",
    "Quantile Regressor",
    "Logistic Regression"
]

CLASSIFICATION_MODELS = [
    "K Nearest Neighbors",
    "Support Vector Machine",
    "Decision Tree"
]


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name.
    Args:
        model_name(str): the name of the model as a string
    Returns:
        Model: a model instance given its str name
    """
    if model_name == "Multiple Linear Regression":
        return MultipleLinearRegression()
    if model_name == "Quantile Regressor":
        return QuantileRegressor()
    if model_name == "Logistic Regression":
        return LogisticRegression()
    if model_name == "K Nearest Neighbors":
        return KNearestNeighbors()
    if model_name == "Support Vector Machine":
        return SupportVectorMachine()
    if model_name == "Decision Tree":
        return DecisionTreeClassifier()
