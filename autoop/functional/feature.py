
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature

def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    features = []

    data = dataset.read()

    # check the first row with the feature names
    for feature_name in data.columns:
        # Check if the feature is numerical
        if data[feature_name].dtype in ['int64', 'float64']:
            features.append(Feature(name=feature_name, type="numerical"))
        else:
            features.append(Feature(name=feature_name, type="categorical"))

    return features
