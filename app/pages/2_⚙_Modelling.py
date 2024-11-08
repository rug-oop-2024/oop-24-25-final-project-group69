import pickle
import streamlit as st
import pandas as pd

from io import BytesIO

from app.core.system import AutoMLSystem
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.model.__init__ import (
    Model, REGRESSION_MODELS, CLASSIFICATION_MODELS, get_model
)
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.metric import METRICS, get_metric, Metric


def write_helper_text(text: str) -> None:
    """
    Displays helper text in the Streamlit app with a gray color.

    Args:
        text (str): The text to display.
    """
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


def select_features(features: list[Feature]) -> tuple[list[Feature],
                                                      Feature,
                                                      str] | None:
    """
    Allows the user to select input features and a target feature.

    Args:
        features (list[Feature]): List of feature objects.

    Returns:
        tuple or None: Selected input features, target feature, task type.
    """
    feature_names = {feature.name: feature for feature in features}

    selected_input_features = st.multiselect(
        "Select an input feature",
        list(feature_names),
    )

    if len(selected_input_features) > len(feature_names) - 1:
        st.error(f"Maximum {len(feature_names) - 1} features.")
    elif len(selected_input_features) == len(feature_names) - 1:
        selected_target_feature = [
            name for name in feature_names if(
                name not in selected_input_features
                )
        ][0]

        st.write(f"Input features: {selected_input_features}")
        st.write(f"Target feature: {selected_target_feature}")

        selected_input_features = [
            feature_names[name] for name in selected_input_features
        ]
        selected_target_feature = feature_names[selected_target_feature]

        if selected_target_feature.type == "numerical":
            task_type = "regression"
        else:
            task_type = "classification"
        st.write(f"Task type: {task_type}")

        return (selected_input_features, selected_target_feature, task_type)

    return None


def select_model(task_type: str) -> Model:
    """
    Allows the user to select a model based on the task type.

    Args:
        task_type (str): The type of task ("regression" or "classification").

    Returns:
        Model: Selected model object.
    """
    model_name = (
        st.selectbox("Select a model", REGRESSION_MODELS)
        if task_type == "regression"
        else st.selectbox("Select a model", CLASSIFICATION_MODELS)
    )

    return get_model(model_name)


def select_split() -> float:
    """
    Prompts the user to enter a data split value.

    Returns:
        float: Split value entered by the user.
    """
    return st.number_input("Enter split", min_value=0.01, max_value=0.99,
                           step=0.01, format="%.2f")


def select_metrics(task_type: str) -> list[Metric]:
    """
    Allows the user to select evaluation metrics.

    Args:
        task_type (str): The type of task ("regression" or "categorical").

    Returns:
        list[Metric]: List of selected metrics.
    """
    selected_names = (
        st.multiselect("Select a metric", METRICS[:3])
        if task_type == "regression"
        else st.multiselect("Select a metric", METRICS[3:])
    )

    return [get_metric(name) for name in selected_names]


def summary(pipeline: Pipeline) -> None:
    """
    Displays the summary of the pipeline.

    Args:
        pipeline (Pipeline): The pipeline object.
    """
    st.write("Pipeline summary:", pipeline)


def train(pipeline: Pipeline) -> None:
    """
    Trains the pipeline and displays metrics and predictions.

    Args:
        pipeline (Pipeline): The pipeline object.
    """
    st.write("Model metrics and predictions:", pipeline.execute())


def serialize_data(pipeline: Pipeline) -> bytes:
    """
    Serializes the pipeline's artifacts, model, and metrics.

    Args:
        pipeline (Pipeline): The pipeline object.

    Returns:
        bytes: Serialized data as bytes.
    """
    artifacts_data = {
        artifact.name: artifact.data for artifact in pipeline.artifacts
    }
    artifacts_data.update({"model": pipeline.model,
                           "metrics": pipeline.metrics})
    return pickle.dumps(artifacts_data)


def save(pipeline: Pipeline) -> None:
    """
    Prompts the user to enter a name and version to save the pipeline.

    Args:
        pipeline (Pipeline): The pipeline object.
    """
    name = st.text_input("Enter a name for the pipeline:")
    version = st.text_input("Enter a version for the pipeline:")

    if name and version:
        artifact = Artifact(
            type="pipeline",
            name=name,
            version=version,
            asset_path=f"{name}_pipeline",
            data=serialize_data(pipeline)
        )
        automl.registry.register(artifact)


st.set_page_config(page_title="Modelling", page_icon="📈")

st.write("# ⚙ Modelling")
write_helper_text("Design a machine learning pipeline to train a model.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

datasets_by_name = {dataset.name: dataset for dataset in datasets}

if datasets_by_name:
    selected_name = st.selectbox("Select a dataset", list(datasets_by_name))
    selected_dataset = datasets_by_name[selected_name]
    csv_data = automl.registry._storage.load(selected_dataset.asset_path)
    df = pd.read_csv(BytesIO(csv_data))
    recreated_dataset = Dataset.from_dataframe(df, selected_dataset.name,
                                               selected_dataset.asset_path)

    features = detect_feature_types(recreated_dataset)
    selection = select_features(features)

    if selection:
        input_features, target_feature, task_type = selection
        model = select_model(task_type)
        split = select_split()
        metrics = select_metrics(task_type)

        if split and metrics:
            pipeline = Pipeline(metrics, recreated_dataset, model,
                                input_features, target_feature, split)
            summary(pipeline)
            train(pipeline)
            save(pipeline)
    else:
        st.write("You have not selected all input features yet...")
