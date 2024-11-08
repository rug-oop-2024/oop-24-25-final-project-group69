import pickle
import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.pipeline import Pipeline


def load(pipelines: list[Artifact]) -> Pipeline:
    """
    Loads a selected pipeline from the available list of artifacts.

    Args:
        pipelines (list[Artifact]): List of pipeline artifacts.

    Returns:
        Pipeline: The recreated pipeline object.
    """
    pipelines_by_names = {pipeline.name: pipeline for pipeline in pipelines}

    selected_name = st.selectbox("Select a pipeline", list(pipelines_by_names))
    selected_pipeline = pipelines_by_names[selected_name]

    artifact_data = automl.registry._storage.load(selected_pipeline.asset_path)
    deserialized_data = pickle.loads(artifact_data)

    # Extract and recreate the pipeline configuration
    pipeline_config = pickle.loads(deserialized_data['pipeline_config'])

    recreated_pipeline = Pipeline(
        metrics=deserialized_data["metrics"],
        dataset=None,
        model=deserialized_data["model"],
        input_features=pipeline_config['input_features'],
        target_feature=pipeline_config['target_feature'],
        split=pipeline_config['split']
    )

    st.write("Pipeline summary:", recreated_pipeline)
    return recreated_pipeline


def predict(pipeline: Pipeline) -> dict:
    """
    Prompts the user to upload a CSV file and executes predictions.

    Args:
        pipeline (Pipeline): The pipeline object to use for predictions.

    Returns:
        dict: Dictionary of prediction results.
    """
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        file_name = uploaded_file.name.split('.')[0]
        file_path = uploaded_file.name

        dataset = Dataset.from_dataframe(data=df, name=file_name,
                                         asset_path=file_path)

        pipeline.dataset = dataset
        return pipeline.execute()


st.set_page_config(page_title="Deployment", page_icon="⬇")

automl = AutoMLSystem.get_instance()

pipelines = automl.registry.list(type="pipeline")

if pipelines:
    selected_pipeline = load(pipelines)

    predictions = predict(selected_pipeline)

    if predictions:
        st.write(predictions)
