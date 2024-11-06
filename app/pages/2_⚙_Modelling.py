import pickle
import streamlit as st
import pandas as pd

from io import BytesIO
from pathlib import Path

from app.core.system import AutoMLSystem
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.model.__init__ import Model, REGRESSION_MODELS, CLASSIFICATION_MODELS, get_model
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.metric import METRICS, get_metric, Metric, MeanAbsoluteError, MicroPrecision, MicroRecall, RSquared, Accuracy, MeanSquaredError


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str) -> None:
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


def select_features(features: list[Feature]) -> tuple[list[Feature], Feature, str] | None:
    feature_names = {feature.name: feature for feature in features}

    selected_input_features = st.multiselect(
        "Select an input feature",
        list(feature_names),
    )

    if len(selected_input_features) > len(feature_names) - 1:
        st.error(f"You can select a maximum of {len(feature_names) - 1} features.")
  
    elif len(selected_input_features) == len(feature_names) - 1:
        selected_target_feature = [feature_name for feature_name in feature_names if feature_name not in selected_input_features][0]

        selected_input_features = [feature_names[selected_input_feature] for selected_input_feature in selected_input_features]
        selected_target_feature = feature_names[selected_target_feature]

        if selected_target_feature.type == "numerical":
            task_type = "regression"
        else:
            task_type = "categorical"
            
        st.write(f"The task type is {task_type}")
        return (selected_input_features, selected_target_feature, task_type)
  
    return None

def select_model(task_type: str) -> Model:
    if task_type == "regression":
        # select model
        selected_model_name = st.selectbox("Select a model", REGRESSION_MODELS)      
    else:
        selected_model_name = st.selectbox("Select a model", CLASSIFICATION_MODELS)

    return get_model(selected_model_name)


def select_split() -> float:
    return st.number_input("Enter split", min_value=0.0, max_value=1.0, step=0.01, format="%.2f")


def select_metrics(task_type: str) -> list[Metric]:
    if task_type == "regression":
        # select model
        selected_metrics_names = st.multiselect("Select a metric", METRICS[:3])    
    else:
        selected_metrics_names = st.multiselect("Select a metric", METRICS[3:])
    
    return [get_metric(metric_name) for metric_name in selected_metrics_names]


def summary(pipeline: Pipeline) -> None:
    st.write(f"This is your beautifully formatted pipeline: {pipeline}")


def train(pipeline: Pipeline) -> None:
    st.write(f"These are the metrics and predictions of your trained model: {pipeline.execute()}")


def save(pipeline: Pipeline) -> None:
    selected_pipeline_name = st.text_input("Enter a name for the pipeline:")
    selected_pipeline_version = st.text_input("Enter a version for the pipeline:")

    pipeline_artifacts = pipeline.artifacts
    pipeline_artifacts_data = {}
    for pipeline_artifact in pipeline_artifacts:
        pipeline_artifacts_data[pipeline_artifact.name] = pipeline_artifact.data

    serialized_data = pickle.dumps(pipeline_artifacts_data)

    artifact_pipeline = Artifact(
        type="pipeline",
        name=selected_pipeline_name,
        version=selected_pipeline_version,
        asset_path="pipeline",
        data=serialized_data
    )

    automl.registry.register(artifact_pipeline)
    


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# your code here
datasets_by_name = {dataset.name: dataset for dataset in datasets}


if datasets_by_name is not None:
    # Use Streamlit's selectbox to allow the user to select a dataset
    selected_dataset_name = st.selectbox("Select a dataset", list(datasets_by_name))
    selected_dataset = datasets_by_name[selected_dataset_name]
    csv_data_from_storage = automl.registry._storage.load(selected_dataset.asset_path)
    df = pd.read_csv(BytesIO(csv_data_from_storage))
    recreated_dataset = Dataset.from_dataframe(
        data=df,
        name=selected_dataset.name,
        asset_path=selected_dataset.asset_path
    )
    features = detect_feature_types(recreated_dataset)

    selection = select_features(features)
    if selection:
        selected_input_features, selected_target_feature, task_type = selection
        selected_model = select_model(task_type)
        selected_split = select_split()
        selected_metrics = select_metrics(task_type)

        if selected_split and selected_metrics is not None:
            pipeline = Pipeline(selected_metrics, recreated_dataset, selected_model, selected_input_features, selected_target_feature, selected_split)
            summary(pipeline)
            train(pipeline)

            save(pipeline)

    else:
        st.write("You have yet not selected all input features...")