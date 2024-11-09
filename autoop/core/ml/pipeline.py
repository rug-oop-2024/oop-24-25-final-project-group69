from typing import List
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline:
    """
    A class to manage the machine learning pipeline, including data
    preprocessing, model training, evaluation, and artifact management.

    Attributes:
        metrics (List[Metric]): A list of metrics used for evaluation.
        dataset (Dataset): The dataset to be used in the pipeline.
        model (Model): The model to be trained.
        input_features (List[Feature]): A list of input features for the model.
        target_feature (Feature): The target feature for prediction.
        split (float): The proportion of the dataset to use for training.
    """

    def __init__(self,
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split: float = 0.8) -> None:
        """
        Initializes the Pipeline with the specified metrics, dataset, model,
        input features, target feature, and data split ratio.

        Args:
            metrics (List[Metric]): List of metrics for evaluating the model.
            dataset (Dataset): The dataset to use for training and testing.
            model (Model): The model to be trained on the dataset.
            input_features (List[Feature]): List of input features.
            target_feature (Feature): The target feature for the prediction.
            split (float): Proportion of data to use for training
            (default: 0.8).

        Raises:
            ValueError: If the target feature type does not match
            the model type.
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split

        is_categorical = target_feature.type == "categorical"
        is_not_classification = model.type != "classification"

        if is_categorical and is_not_classification:
            raise ValueError("Model type must be classification"
                             "for categorical "
                             "target feature")
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError("Model type must be regression for continuous "
                             "target feature")

    def __str__(self) -> str:
        """
        Returns a string representation of the Pipeline instance, showing
        key attributes.

        Returns:
            str: String representation of the Pipeline.
        """
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def dataset(self) -> Dataset:
        """
        Retrieves the dataset associated with the pipeline.

        Returns:
            Dataset: The dataset instance used in the pipeline.
        """
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: Dataset) -> None:
        """
        Sets the dataset for the pipeline.

        Args:
            dataset (Dataset): The dataset instance to be used in the pipeline.

        Raises:
            ValueError: If the provided object is not of type Dataset.
        """
        if isinstance(dataset, Dataset):
            self._dataset = dataset
        else:
            raise ValueError("Object is not of type Dataset")

    @property
    def metrics(self) -> list[Metric]:
        """
        Gets the metrics associated with the Pipeline.

        Returns:
            Model: The metrics instance used in the pipeline.
        """
        return self._metrics

    @property
    def model(self) -> Model:
        """
        Gets the model associated with the Pipeline.

        Returns:
            Model: The model instance used in the pipeline.
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Generates a list of artifacts associated with the pipeline,
        including feature preprocessors and the trained model.

        Returns:
            List[Artifact]: List of artifacts for the pipeline.
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(name="pipeline_config",
                                  data=pickle.dumps(pipeline_data)))
        artifacts.append(self.model.to_artifact(
            name=f"pipeline_model{self._model.type}"))
        return artifacts

    def _register_artifact(self,
                           name: str,
                           artifact: Artifact) -> None:
        """
        Registers an artifact within the pipeline for later retrieval.

        Args:
            name (str): The name of the artifact.
            artifact: The artifact to register.
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """
        Preprocesses the features of the dataset, registering artifacts
        for both input and target features.
        """
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset)[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(self._input_features,
                                            self._dataset)
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        self._output_vector = target_data
        self._input_vectors = [data for (feature_name, data, artifact)
                               in input_results]

    def _split_data(self) -> None:
        """
        Splits the dataset into training and testing sets based on the
        specified split ratio.
        """
        split = self._split
        self._train_X = [
            vector[:int(split * len(vector))] for vector in self._input_vectors
        ]
        self._test_X = [
            vector[int(split * len(vector)):] for vector in self._input_vectors
        ]
        self._train_y = self._output_vector[
            :int(split * len(self._output_vector))]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)):]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """
        Concatenates a list of numpy arrays into a single 2D array.

        Args:
            vectors (List[np.array]): List of numpy arrays to concatenate.

        Returns:
            np.array: A single concatenated numpy array.
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """
        Trains the model using the training dataset after compacting the
        input vectors.
        """

        X = self._compact_vectors(self._train_X)
        Y = self._train_y

        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """
        Evaluates the model using the test dataset, calculating metrics
        for model performance.
        """
        X = self._compact_vectors(self._test_X)
        Y = self._test_y

        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((str(metric), float(result)))
        self._predictions = predictions

    def execute(self) -> dict:
        """
        Executes the full pipeline process, including feature preprocessing,
        data splitting, model training, and evaluation, returning metrics
        and predictions.

        Returns:
            dict: A dictionary containing training metrics, test metrics,
                  and predictions.
        """
        self._preprocess_features()
        self._split_data()
        self._train()

        self._evaluate()
        test_metrics_results = self._metrics_results
        test_predictions = self._predictions

        original_test_X, original_test_y = self._test_X, self._test_y
        # the part of the dataset that was used for training is now tested
        self._test_X, self._test_y = self._train_X, self._train_y
        self._evaluate()
        train_metrics_results = self._metrics_results
        # set the test data back to the original values
        self._test_X, self._test_y = original_test_X, original_test_y

        return {
            "train_metrics": train_metrics_results,
            "test_metrics": test_metrics_results,
            "predictions": test_predictions
        }
