from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.lasso_regression import MultipleLinearRegression
from autoop.core.ml.model.regression.ard_regression import ARDRegression
from autoop.core.ml.model.regression.ridge_regression import RidgeRegression
from autoop.core.ml.model.classification.decision_tree import DecisionTreeClassifier
from autoop.core.ml.model.classification.knn import KNearestNeighbors
from autoop.core.ml.model.classification.random_forest_classifier import RandomForestClassifier

REGRESSION_MODELS = [
    "Lasso Regression",
    "ARD Regression",
    "Ridge Regression"
]

CLASSIFICATION_MODELS = [
    "K Nearest Neighbors",
    "Random Forest Classifier",
    "Decision Tree"
]


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name.
    Args:
        model_name(str): the name of the model as a string
    Returns:
        Model: a model instance given its str name
    """
    if model_name == "Lasso Regression":
        return MultipleLinearRegression()
    if model_name == "ARD Regression":
        return ARDRegression()
    if model_name == "Ridge Regression":
        return RidgeRegression()
    if model_name == "K Nearest Neighbors":
        return KNearestNeighbors()
    if model_name == "Random Forest Classifier":
        return RandomForestClassifier()
    if model_name == "Decision Tree":
        return DecisionTreeClassifier()
