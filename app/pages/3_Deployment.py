import ast
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


# this has to be generalized
def load(pipelines: list[Artifact]) -> Pipeline:
    pipelines_by_names = {pipeline.name: pipeline for pipeline in pipelines}

    selected_pipeline_name = st.selectbox("Select a pipeline", list(pipelines_by_names))
    selected_pipeline = pipelines_by_names[selected_pipeline_name]

    artifact_data_from_storage = automl.registry._storage.load(selected_pipeline.asset_path)
    deserialized_data = pickle.loads(artifact_data_from_storage)

    model_data = pickle.loads(deserialized_data['pipeline_modelregression'])
    model = get_model("Multiple Linear Regression")
    model._parameters = {
        "hyperparameters": model_data['hyperparameters'],
        "strict parameters": model_data['strict parameters']
    }

    # Access and set up pipeline configuration details
    pipeline_config = pickle.loads(deserialized_data['pipeline_config'])

    # Recreate the pipeline with the deserialized config and model
    recreated_pipeline = Pipeline(
        metrics=[get_metric("mean_squared_error")],
        dataset=None,
        model=model,
        input_features=pipeline_config['input_features'],
        target_feature=pipeline_config['target_feature'],
        split=pipeline_config['split']
    )

    # Display pipeline for verification
    st.write(f"This is your formatted pipeline: {recreated_pipeline}")
    return recreated_pipeline


def predict(pipeline: Pipeline) -> dict:
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        dataset = Dataset.from_dataframe(
            data=df,
            name="iris",
            asset_path="iris.csv"
        )
        
        pipeline._dataset = dataset

        return pipeline.execute()


st.set_page_config(page_title="Deployment", page_icon="📈")

automl = AutoMLSystem.get_instance()

pipelines = automl.registry.list(type="pipeline")


if pipelines is not None:
    selected_pipeline = load(pipelines)

    predictions = predict(selected_pipeline)
    
    if predictions is not None:
        st.write(predictions)
