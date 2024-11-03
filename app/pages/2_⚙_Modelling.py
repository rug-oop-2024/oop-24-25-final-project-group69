import streamlit as st
import pandas as pd

from io import BytesIO
from pathlib import Path

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.model.__init__ import REGRESSION_MODELS, CLASSIFICATION_MODELS, get_model
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.metric import METRICS, get_metric, Metric, MeanAbsoluteError, MicroPrecision, MicroRecall, RSquared, Accuracy, MeanSquaredError


st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

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
    
    feature_names = {feature.name: feature for feature in features}

    selected_input_features = st.multiselect("Select an input feature", list(feature_names))

    if len(selected_input_features) > len(feature_names) - 1:
        st.error(f"You can select a maximum of {len(feature_names) - 1} features.")

    elif len(selected_input_features) == len(feature_names) - 1:
        selected_target_feature = st.selectbox("Select a target feature",
                                            [feature_name for feature_name in feature_names if feature_name not in selected_input_features])

        selected_input_features = [feature_names[selected_input_feature] for selected_input_feature in selected_input_features]
        selected_target_feature = feature_names[selected_target_feature]

        selected_model = None

        if selected_target_feature.type == "numerical":
            st.write("The task type is regression")

            # select model
            selected_model_name = st.selectbox("Select a model", REGRESSION_MODELS)
            selected_model = get_model(selected_model_name)

            # get corresponding metrics
            possible_metrics_names = METRICS[:3]
            
        else:
            st.write("The task type is classification")

            selected_model_name = st.selectbox("Select a model", CLASSIFICATION_MODELS)
            selected_model = get_model(selected_model_name)

            possible_metrics_names = METRICS[3:]

        selected_split = st.number_input("Enter split", min_value=0.0, max_value=1.0, step=0.01, format="%.2f")

        selected_metrics_names = st.multiselect("Select a metric", list(possible_metrics_names))

        selected_metrics = [get_metric(metric_name) for metric_name in selected_metrics_names]

        if selected_split and selected_metrics is not None:

            pipeline = Pipeline(selected_metrics, recreated_dataset, selected_model, selected_input_features, selected_target_feature, selected_split)
            st.write(f"This is your beautifully formatted pipeline: {pipeline}")

            st.write(f"These are the metrics and predictions of your trained model: {pipeline.execute()}")
